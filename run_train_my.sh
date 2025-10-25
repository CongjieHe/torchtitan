NGPU=${NGPU:-"1"}
LOG_RANK=0
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/qwen3-moe/train_configs/qwen3_30b_a3b.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/gpt_oss/train_configs/gpt_oss_20b.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}