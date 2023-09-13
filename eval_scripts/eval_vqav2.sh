#!/bin/bash 

image_dir_path=$1
question_path=$2
annotation_path=$3
ft_model_path=$4

cd ..

if [ $# -eq 4 ]; then
    echo "Evaluating finetuned MLLM."
	python llava/eval/mmoverall/evaluate.py --eval_vqav2 --additional_shots 3 --vqav2_image_dir_path ${image_dir_path} \
		--vqav2_questions_json_path ${question_path} --vqav2_annotations_json_path ${annotation_path} \
		--batch_size 1 --top_k 50 --precision fp32 --num_samples 5000 --num_trials 1 \
		--trial_seeds 42 --device 0 --num_beams 1 --do_sample --conv_mode v0 --model_path ${ft_model_path}

elif [ $# -eq 5 ]; then
    echo "Evaluating lora tuned MLLM."
	lora_model_path=$5
	python llava/eval/mmoverall/evaluate.py --eval_vqav2 --additional_shots 3 --vqav2_image_dir_path ${image_dir_path} \
		--vqav2_questions_json_path ${question_path} --vqav2_annotations_json_path ${annotation_path} \
		--batch_size 1 --top_k 50 --precision fp32 --num_samples 5000 --num_trials 1 \
		--trial_seeds 42 --device 0 --num_beams 1 --do_sample --conv_mode v0 --model_path ${lora_model_path} \
		--model_base ${ft_model_path}
else
    echo "Wrong argument number."
fi