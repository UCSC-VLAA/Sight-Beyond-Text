#!/bin/bash 

image_dir_path=$1
test_annot_path=$2
demo_annot_path=$3
ft_model_path=$4

cd ..

if [ $# -eq 4 ]; then
    echo "Evaluating finetuned MLLM."
	python llava/eval/mmoverall/evaluate.py --eval_flickr30 --additional_shots 3 --flickr_image_dir_path ${image_dir_path} \
		--flickr_annotations_json_path ${test_annot_path} --flickr_demo_annotations_json_path ${demo_annot_path} \
		--batch_size 1 --top_k 50 --precision fp32 --num_trials 1 --length_penalty 1.0 \
		--trial_seeds 42 --device 0 --num_beams 1 --do_sample --conv_mode v0 --model_path ${ft_model_path}

elif [ $# -eq 5 ]; then
    echo "Evaluating lora tuned MLLM."
	lora_model_path=$5
	python llava/eval/mmoverall/evaluate.py --eval_flickr30 --additional_shots 3 --flickr_image_dir_path ${image_dir_path} \
		--flickr_annotations_json_path ${test_annot_path} --flickr_demo_annotations_json_path ${demo_annot_path} \
		--batch_size 1 --top_k 50 --precision fp32 --num_trials 1 --length_penalty 1.0 \
		--trial_seeds 42 --device 0 --num_beams 1 --do_sample --conv_mode v0 --model_path ${lora_model_path} \
		--model_base ${ft_model_path}
else
    echo "Wrong argument number."
fi