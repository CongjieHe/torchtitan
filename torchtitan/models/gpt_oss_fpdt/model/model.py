# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from packaging import version 

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

import deepspeed.comm as dist
from deepspeed.utils import groups
from deepspeed.sequence.fpdt_layer import _FPDTGPUOffloadingAttentionImpl_, FPDT_InputConstruct

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    flash_attn_version = version.parse(flash_attn.__version__)
except ImportError:
    _flash_attn_forward = None
    _flash_attn_backward = None

from .args import GptOssModelArgs
from .moe import GptOssMoE


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FPDT_Attention(nn.Module):
    """
    FPDT attention module.
    """
    def __init__(self,
                 config,
                 enable_offloading=True) -> None:
        super(FPDT_Attention, self).__init__()
        if _flash_attn_forward is None or _flash_attn_backward is None:
            raise ImportError(
                "DeepSpeed FPDT requires flash-attn 2.6.3. Please install it with `pip install flash-attn --no-build-isolation`."
            )
        self.config = config
        self.chunk_size = config['chunk_size']
        self.enable_offloading = enable_offloading

        self.spg = groups._get_sequence_parallel_group()
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        self.word_size = dist.get_world_size(self.spg)

        self.projection_size = config['head_dim'] * config['num_attention_heads']
        self.dim_per_attention_head = config['num_attention_heads'] * config['head_dim']
        self.kv_projection_size = config['head_dim'] * config['num_key_value_heads']
        self.dim = config['dim']

        class args:

            def __init__(self):
                self.ds_sequence_parallel_fpdt_chunk_size = config['chunk_size']

        self.args = args()

        if config['dtype'] == 'bf16':
            self.dtype = torch.bfloat16
        elif config['dtype'] == 'fp8':
            self.dtype = torch.float8_e4m3fn
        else:
            raise ValueError(f"Unsupported dtype: {config['dtype']}")


        if dist.get_rank() == 0:
            self.qkv_linear_weight = torch.nn.Parameter(
                torch.empty(self.dim + 2 * self.dim, self.dim, device=dist.get_rank(), dtype=self.dtype))
            torch.nn.init.normal_(self.qkv_linear_weight, mean=0.0, std=0.02)

            self.qkv_linear_bias = torch.nn.Parameter(torch.empty(self.dim + 2 * self.dim, device=dist.get_rank(), dtype=self.dtype))
            torch.nn.init.normal_(self.qkv_linear_bias, mean=0.0, std=0.02)
        else:
            self.qkv_linear_weight = torch.nn.Parameter(
                torch.empty(self.dim + 2 * self.dim, self.dim, device=dist.get_rank(), dtype=self.dtype))
            self.qkv_linear_bias = torch.nn.Parameter(torch.empty(self.dim + 2 * self.dim, device=dist.get_rank(), dtype=self.dtype))

        dist.broadcast(self.qkv_linear_weight, src=0, group=self.spg)
        dist.broadcast(self.qkv_linear_bias, src=0, group=self.spg)

    def forward(self,
                x,
                rotary_pos_emb,
                attention_mask) -> torch.Tensor:

        fpdt_input_tensor = FPDT_InputConstruct(x, None, None, None, None, self.args,
                                                self.word_size, dist.get_rank()).generate()[0].permute(1, 0, 2)
        self.num_chunks_attn = fpdt_input_tensor.shape[0] * self.word_size // self.chunk_size

        output = _FPDTGPUOffloadingAttentionImpl_.apply(
            fpdt_input_tensor, None, None, None, self.spg, 2, 0, self.dim,
            self.projection_size, self.dim_per_attention_head, self.kv_projection_size,
            self.qkv_linear_weight, self.qkv_linear_bias, 0, self.num_chunks_attn, self.enable_offloading
        )

        return output.flatten(2).contiguous()

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: GptOssModelArgs):

        super().__init__()
        self.use_sliding_attention = layer_id % 2 == 0
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe = GptOssMoE(
            model_args, dim=model_args.dim, hidden_dim=model_args.moe_inter_dim
        )
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        self.spg = groups._get_sequence_parallel_group()

        self.config = {
            'chunk_size': 128,
            'head_dim': model_args.head_dim,
            'num_attention_heads': model_args.n_heads,
            'num_key_value_heads': model_args.n_kv_heads,
            'dim': model_args.dim,
            'dtype': model_args.dtype,
        }

        self.attention = FPDT_Attention(self.config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # Extract the appropriate mask for this layer
        if self.use_sliding_attention:
            layer_mask = attention_masks.get("sliding_window_mask", None)
        else:
            layer_mask = attention_masks.get("basic_mask", None)
        assert layer_mask is not None

        x = x + self.attention(self.attention_norm(x), rope_cache, layer_mask)
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        # self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


class GptOssModel(nn.Module, ModelProtocol):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: GptOssModelArgs):
        super().__init__()
        self.model_args = model_args
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args).to(
                torch.bfloat16
            )

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:

        basic_mask_mods = []
        sliding_window_mask_mods = [
            get_sliding_window_mask_mod(self.model_args.sliding_window_size)
        ]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
                basic_mask_mods.append(get_causal_mask_mod())
            case "block_causal":
                B = input_batch.shape[0]
                basic_mask_mods.append(
                    get_document_mask_mod(input_batch, tokenizer.eos_id)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )

        # create basic attention mask: causal or block_causal
        basic_mask = create_attention_mask(
            and_masks(*basic_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        # create sliding window mask, has to be created on top of basic attention mask
        sliding_window_mask = create_attention_mask(
            and_masks(*basic_mask_mods, *sliding_window_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        return {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks)
        h = self.norm(h)
        output = self.output(h)
        return output