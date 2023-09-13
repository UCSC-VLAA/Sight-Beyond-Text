#!/bin/bash 

mmlu_data=$1
base_model=$2

cd ..

if [ $# -eq 2 ]; then
    echo "Evaluating finetuned MLLM on MMLU."
	python ./llava/eval/mmlu/evaluate_mmlu.py main --model_path ${base_model} \
                                --model_name llama --data_dir ${mmlu_data} \
                                --out_file ./mmlu_results/test_finetuned_mmlu.json
elif [ $# -eq 3 ]; then
    echo "Evaluating lora tuned MLLM on MMLU."
	peft_model=$3
	python ./llava/eval/mmlu/evaluate_mmlu.py main --model_path ${base_model} --lora_path ${peft_model} \
                                --model_name llama --data_dir ${mmlu_data} \
                                --out_file ./mmlu_results/test_lora_mmlu.json
else
    echo "Wrong argument number."
fi