# NLP Evaluations

For NLP ability evaluations except MMLU, we directly employ the  benchmark toolbox [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Make sure you follow their guidelines to install the toolbox, and run these commands for one whole NLP evaluation (except MMLU).

For finetuned MLLMs:

```bash
export ft_model_path=finetuning-repo-id-hf
cd ../../eval_scripts
bash eval_nlp.sh /your/path/to/lm-evaluation-harness ft_model_path
```

For lora-tuned MLLMs:

```bash
export base_model_path=meta-llama/llama-2-7b-chat-hf
export lora_model=lora-repo-id-hf
cd ../../eval_scripts
bash eval_nlp.sh /your/path/to/lm-evaluation-harness base_model_path lora_model
```

For MMLU evaluation, first download data [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar). Then evaluate the model:

```bash
export ft_model_path=finetuning-repo-id-hf
export base_model_path=meta-llama/llama-2-7b-chat-hf
export lora_model=lora-repo-id-hf

cd ../../eval_scripts
## For finetuned MLLMs
bash eval_mmlu.sh /path/to/mmlu_data ft_model_path

## For lora-tuned MLLMs
bash eval_mmlu.sh /path/to/mmlu_data base_model_path lora_model
```

# Multi-Modal Evaluations

This directory contains end-to-end pipelines for evaluations on seven multi-modal benchmarks (including two tasks with corrupted images) of our trained MLLMs. We will introduce the evaluation pipeline and the data downloading guides in this document. First, define the model:

```bash
export ft_model_path=finetuning-repo-id-hf
export base_model_path=meta-llama/llama-2-7b-chat-hf
export lora_model=lora-repo-id-hf
cd ../../eval_scripts
```

## VQAv2

### Dataset

Download the Val [images](https://cocodataset.org/#download) and Val [annotations](https://visualqa.org/download.html) for evaluation.

### Running Scripts

```bash
## For finetuned MLLMs
bash eval_vqav2.sh /path/to/image /path/to/vqa_question /path/to/vqa_annotation ft_model_path

## For lora-tuned MLLMs
bash eval_vqav2.sh /path/to/image /path/to/vqa_question /path/to/vqa_annotation base_model_path lora_model
```

For evaluating on corrputed images, just simply replace the image directory path (`/path/to/image`).

## MSCOCO

### Dataset

Download the post-processed data from [here](https://drive.google.com/file/d/1J922lIqzXpLfqfWd2-F3ZI3mW59lqlBu/view?usp=sharing). The resulting MSCOCO Captioning data should look like this:
```bash
.
├── ./mscoco/                    
    ├── mscoco_train.json # Contains the training set text captions of MSCOCO
    ├── mscoco_val.json # Contains the validation set text captions of MSCOCO
    ├── mscoco_test.json # Contains the test set text captions of MSCOCO
    └── test_images # Contains the test set images of MSCOCO
```

### Running Scripts

```bash
## For finetuned MLLMs
bash eval_mscoco.sh /path/to/test_images /path/to/mscoco_test.json /path/to/mscoco_train.json ft_model_path

## For lora-tuned MLLMs
bash eval_mscoco.sh /path/to/test_images /path/to/mscoco_test.json /path/to/mscoco_train.json base_model_path lora_model
```

## Flickr30K

### Dataset

Download the post-processed data from [here](https://drive.google.com/file/d/1i8-v-U3qlhK9uW_RzV3iV8IRJlKTpcBZ/view?usp=sharing). The resulting Flcikr30K Captioning data should look like this:
```bash
.
├── ./flickr30k/                    
    ├── flickr30k_train.json # Contains the training set text captions of Flickr30k
    ├── flickr30k_val.json # Contains the validation set text captions of Flickr30k
    ├── flickr30k_test.json # Contains the test set text captions of Flickr30k
    └── test_images # Contains the test set images of Flickr30k
```

### Running Scripts

```bash
## For finetuned MLLMs
bash eval_mscoco.sh /path/to/test_images /path/to/flickr30k_test.json /path/to/flickr30k_train.json ft_model_path

## For lora-tuned MLLMs
bash eval_mscoco.sh /path/to/test_images /path/to/flickr30k_test.json /path/to/flickr30k_train.json base_model_path lora_model
```

## MME Benchmark

### Dataset

Download the dataset follow instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

### Running Scripts

```bash
## For finetuned MLLMs
bash eval_mme.sh /path/to/mme ft_model_path

## For lora-tuned MLLMs
bash eval_mme.sh /path/to/mme base_model_path lora_model
```

## POPE Benchmark

### Dataset

Download and post-process the dataset follow instructions [here](https://github.com/RUCAIBox/POPE).

### Running Scripts

```bash
## For finetuned MLLMs
bash eval_pope.sh /path/to/mscoco/val2014 /path/to/pope/annotations ft_model_path

## For lora-tuned MLLMs
bash eval_pope.sh /path/to/mscoco/val2014 /path/to/pope/annotations base_model_path lora_model
```