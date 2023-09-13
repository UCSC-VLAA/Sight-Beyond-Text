#!/bin/bash 

lm_eval_path=$1
base_model=$2

cd lm_eval_path || exit


if [ $# -eq 2 ]; then
    echo "Evaluating finetuned MLLM on NLP benchmarks."
	python main.py --model hf-causal-experimental --model_args pretrained=${base_model} --no_cache --tasks ethics_cm,ethics_virtue,ethics_justice,ethics_utilitarianism,ethics_deontology --device cuda:0 --batch_size auto --num_fewshot 0 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model} --no_cache --tasks gsm8k --device cuda:0 --batch_size auto --num_fewshot 8 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model} --no_cache --tasks mathqa --device cuda:0 --batch_size auto --num_fewshot 4 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model} --no_cache --tasks boolq,squad2 --device cuda:0 --batch_size auto --num_fewshot 0 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model} --no_cache --tasks truthfulqa_gen,truthfulqa_mc --device cuda:0 --batch_size auto --num_fewshot 0 &
    wait
elif [ $# -eq 3 ]; then
    echo "Evaluating lora tuned MLLM on NLP benchmarks."
	peft_model=$3
	python main.py --model hf-causal-experimental --model_args pretrained=${base_model},peft=${peft_model} --no_cache --tasks ethics_cm,ethics_virtue,ethics_justice,ethics_utilitarianism,ethics_deontology --device cuda:0 --batch_size auto --num_fewshot 0 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model},peft=${peft_model} --no_cache --tasks gsm8k --device cuda:0 --batch_size auto --num_fewshot 8 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model},peft=${peft_model} --no_cache --tasks mathqa --device cuda:0 --batch_size auto --num_fewshot 4 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model},peft=${peft_model} --no_cache --tasks boolq,squad2 --device cuda:0 --batch_size auto --num_fewshot 0 &

    python main.py --model hf-causal-experimental --model_args pretrained=${base_model},peft=${peft_model} --no_cache --tasks truthfulqa_gen,truthfulqa_mc --device cuda:0 --batch_size auto --num_fewshot 0 &
    wait
else
    echo "Wrong argument number."
fi