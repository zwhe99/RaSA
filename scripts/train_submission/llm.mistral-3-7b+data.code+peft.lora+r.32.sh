set -e
set -u

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"
export TOKENIZERS_PARALLELISM=false


SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORK_DIR=$SCRIPT_DIR/../..
CONFIG_DIR=$WORK_DIR/scripts/config
MODEL_DIR=$WORK_DIR/models
DATA_DIR=$WORK_DIR/data

torchrun --nnodes 1 --nproc_per_node 8 $WORK_DIR/sft.py \
    --base_model mistralai/Mistral-7B-v0.3 \
    --output_dir $MODEL_DIR/llm.mistral-3-7b+data.code+peft.lora+r.32 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 32 \
    --chat_template_name alpaca-chat \
    --data_name ise-uiuc/Magicoder-Evol-Instruct-110K \
    --data_dir $DATA_DIR \
    --train_split train \
    --batch_size 192 \
    --micro_batch_size 2 \
    --num_train_epochs 8 \
    --use_lion \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --save_strategy epoch \
    --bf16 \
    --gc \
    --group_by_length \
    --deepspeed $CONFIG_DIR/deepspeed/deepspeed_config_zero2_lion.json

