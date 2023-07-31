deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id="baichuan_finetune_v3_$(date +%m%d%H)"
project_dir=$(cd "$(dirname $0)"/.; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

echo "output_dir: ${output_dir}"
echo "log_dir: ${log_dir}"


mkdir -p ${output_dir} ${log_dir}

WANDB_MODE=disabled \
deepspeed ${deepspeed_args} \
  promptcblue/supervised_finetuning/fintune.py \
  --finetune_type supervised_finetuning \
  --model_name_or_path baichuan-inc/Baichuan-13B-Base \
  --dataset_name data/PromptCBLUE \
  --preprocessing_num_workers 20 \
  --output_dir ${output_dir} --overwrite_output_dir \
  --num_train_epochs 2 \
  --bf16 \
  --gradient_checkpointing 1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --not_group \
  --per_device_train_batch_size 1 \
  --deepspeed promptcblue/supervised_finetuning/ds_config_zero3.json \
  --validation_split_percentage 5 \
  --logging_steps 20 \
  --save_steps 10000 \
  --do_train \
  --dataloader_num_workers 1 \
  | tee ${log_dir}/train.log \
  2> ${log_dir}/train.err
