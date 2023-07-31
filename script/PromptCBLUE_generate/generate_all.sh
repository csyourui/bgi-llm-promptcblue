#!/bin/bash

model=${1}
project_dir=$(cd "$(dirname $0)"/.; pwd)

dir="${project_dir}/${model}/output"
echo "Path: ${project_dir}"
echo "Model: ${model}"
echo "Dir: ${dir}"

if [ ! -d "$dir" ]
then
    mkdir "$dir"
    echo "Directory $dir created."
else
    echo "Directory $dir already exists."
fi

name="${dir}/out_${model}_med"

for i in {0..7}
do
  nohup ${project_dir}/${model}/generate.sh ${i} ${name} > ${dir}/output_${i}.txt 2>&1 &
done

echo "Start all tasks."
wait
sleep 5
echo "All tasks finished."

cat ${name}_0.json \
    ${name}_1.json \
    ${name}_2.json \
    ${name}_3.json \
    ${name}_4.json \
    ${name}_5.json \
    ${name}_6.json \
    ${name}_7.json > ${project_dir}/${model}/test_predictions.json

wait
echo "Delete temp files"
rm ${name}*.json
