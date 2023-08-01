script_dir=$(cd "$(dirname $0)"/.; pwd)

deepspeed --master_port=$((11000 + $1)) --include localhost:$1\
  ${script_dir}/generate.py \
  --local_rank $1 \
  --output_path_name $2 \
  --model_path model/bgi-promptcblue-baichuan-13b \
  --data_path data/PromptCBLUE \
  --deepspeed ${script_dir}/ds_config.json \
  --max_new_tokens 2048 \
  --temperature 0.7 \
  --inference_batch_size_per_device 1
