import argparse
import json
import sys

import os
import random
import uuid
from collections import defaultdict
from typing import Callable

import more_itertools
import numpy as np
import torch

from eval_datasets import (
    COCOFlickrDataset, 
    VQADataset, 
    PopeDataset,
    MmeDataset, 
    COCOFlickrCaptionDataset
    )

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from mme_metric import get_final_mme_scores
from mme_metric import eval_type_dict as mme_eval_type_dict
from pope_metric import get_pope_results
from coco_metric import compute_caption_metrics, postprocess_captioning_generation
from get_model_output import EvalModel as LLaVAEvalModel

## you can specify the data dir prefix here
DATA_DIR = ""

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    default=os.path.join(CACHE_DIR, "llava-7b"))
parser.add_argument("--model_base", type=str, default=None)

parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=1,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
parser.add_argument(
    "--precision", type=str, default='fp32', choices=['fp16', 'fp32']
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--num_beams",
    type=int,
    default=5,
)
parser.add_argument(
    "--top_k",
    type=int,
    default=10,
)
parser.add_argument(
    "--do_sample",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--max_generation_length",
    type=int,
    default=10,
)
parser.add_argument(
    "--min_generation_length",
    type=int,
    default=1,
)
parser.add_argument(
    "--length_penalty",
    type=float,
    default=-1.0,
)

parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    type=int,
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)
parser.add_argument(
    "--additional_shots",
    type=int,
    default=2,
)


parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)

parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)
parser.add_argument(
    "--eval_pope",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--eval_mme",
    action="store_true",
    default=False,
)

# Dataset arguments
## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default="flickr30k/test_images",
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default="flickr30k/flickr30k_test.json",
)
parser.add_argument(
    "--flickr_demo_annotations_json_path",
    type=str,
    default="flickr30k/flickr30k_train.json",
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    default="mscoco/val2014",
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default="mscoco/mscoco_test.json",
)

parser.add_argument(
    "--coco_demo_annotations_json_path",
    type=str,
    default="mscoco/mscoco_train.json",
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default="mscoco/val2014",
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    ## box image path attached..
    default="vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json",
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default="vqa-v2/v2_mscoco_val2014_annotations.json",
)


DEFAULT_IMAGE_C = [
    'impulse_noise', 'glass_blur',
    'brightness', 'gaussian_blur', 'frost',
    'gaussian_noise', 'speckle_noise', 'defocus_blur',
    'fog', 'spatter', 'saturate', 'zoom_blur', 'shot_noise',
    'snow', 'motion_blur', 'contrast', 'elastic_transform',
    'pixelate', 'jpeg_compression',
]

parser.add_argument(
    "--vqav2_img_mode",
    type=str,
    default="plain",
    choices=DEFAULT_IMAGE_C + ['plain'],
    help="Noise added to images of VQAv2 task.",
)
parser.add_argument(
    "--vqav2_img_noise_severity",
    nargs="+",
    default=[1, 3, 5],
    type=int,
)
parser.add_argument(
    "--caption_img_noise_severity",
    nargs="+",
    default=[1, 3, 5],
    type=int,
)
parser.add_argument(
    "--caption_img_mode",
    default="plain",
    choices=DEFAULT_IMAGE_C + ['plain'],
    type=str,
)
## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default="mscoco/val2014",
)
parser.add_argument(
    "--ok_vqa_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default="OK-VQA/OpenEnded_mscoco_val2014_questions.json",
)
parser.add_argument(
    "--ok_vqa_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default="OK-VQA/mscoco_val2014_annotations.json",
)

## POPE Dataset
parser.add_argument(
    "--pope_annot_dir",
    type=str,
    default=os.path.join(DATA_DIR, "pope/output/coco"),
)
parser.add_argument(
    "--pope_image_dir_path",
    type=str,
    default="mscoco/val2014",
)

## MME Dataset
parser.add_argument(
    "--mme_annot_dir",
    type=str,
    default=os.path.join(DATA_DIR, "mme"),
)

parser.add_argument(
    "--conv_mode",
    type=str,
    default="v0",
)

def setup_seed(seed=3407):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def main():
    args = parser.parse_args()

    ckpt_model = args.model_path.split("/")[-1]
    peft_model = "" if args.model_base is None else args.model_path.split("/")[-1]
    test_results_dir = "test_results"
    generation_config = f"_beam{args.num_beams}-sample{args.do_sample}-k{args.top_k}" if args.eval_vqav2 else ""
    if args.eval_pope:
        task_name = "pope"
        img_noise = ""
    elif args.eval_mme:
        task_name = "mme"
        img_noise = ""
    elif args.eval_vqav2:
        task_name = "vqav2"
        img_noise = "" if args.vqav2_img_mode == "plain" else f"{args.vqav2_img_mode}-{args.vqav2_img_noise_severity}"
    elif args.eval_ok_vqa:
        task_name = "okvqa"
        img_noise = "" if args.vqav2_img_mode == "plain" else f"{args.vqav2_img_mode}-{args.vqav2_img_noise_severity}"
    elif args.eval_coco:
        task_name = "mscoco"
        img_noise = "" if args.caption_img_mode == "plain" else f"{args.caption_img_mode}-{args.caption_img_noise_severity}"
    elif args.eval_flickr30:
        task_name = "flickr30"
        img_noise = "" if args.caption_img_mode == "plain" else f"{args.caption_img_mode}-{args.caption_img_noise_severity}"
    else:
        raise NotImplementedError

    args.results_file = f"./{test_results_dir}/{task_name}_val" \
                        f"{args.num_samples}_{args.num_trials}trials" \
                        f"_shots{args.shots}_{ckpt_model}_{peft_model}_" \
                        f"addshot{args.additional_shots}_{img_noise}{generation_config}.json"

    eval_model = LLaVAEvalModel(args)

    results = defaultdict(list)

    if args.eval_flickr30:
        if args.caption_img_mode != "plain":
            severities = args.caption_img_noise_severity
            noisy_flag = True
            print("Evaluating on Flickr30-C...")
        else:
            severities = [0]
            noisy_flag = False
            print("Evaluating on Flickr30k...")
        scores = []
        for severity in severities:
            image_dir_path = f"flickr30k/test_images-c/{args.caption_img_mode}/{severity}/test_images" \
                if noisy_flag else args.flickr_image_dir_path
            caption_scores = evaluate_coco_flickr(
                eval_model=eval_model,
                batch_size=args.batch_size,
                image_dir_path=image_dir_path,
                annotations_json_path=args.flickr_annotations_json_path,
                demo_annotations_json_path=args.flickr_demo_annotations_json_path,
                num_samples=args.num_samples,
                is_flickr=True,
                additional_shots=args.additional_shots,
                max_generation_length=args.max_generation_length,
                min_generation_length=args.min_generation_length,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                conv_mode=args.conv_mode,
            )
            scores.append(caption_scores)
        results["flickr30"].append(
            {"trials": scores}
        )

    if args.eval_coco:
        if args.caption_img_mode != "plain":
            severities = args.caption_img_noise_severity
            noisy_flag = True
            print("Evaluating on COCO-C...")
        else:
            severities = [0]
            noisy_flag = False
            print("Evaluating on COCO...")
        scores = []
        for severity in severities:
            image_dir_path = f"mscoco/val2014-c/{args.caption_img_mode}/{severity}/val2014" \
                if noisy_flag else args.coco_image_dir_path
            caption_scores = evaluate_coco_flickr(
                eval_model=eval_model,
                batch_size=args.batch_size,
                image_dir_path=image_dir_path,
                annotations_json_path=args.coco_annotations_json_path,
                demo_annotations_json_path=args.coco_demo_annotations_json_path,
                num_samples=args.num_samples,
                additional_shots=args.additional_shots,
                max_generation_length=args.max_generation_length,
                min_generation_length=args.min_generation_length,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                conv_mode=args.conv_mode,
            )
            scores.append(caption_scores)
        results["coco"].append(
            {"trials": scores}
        )

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            output_score_detials = False
            scores, perques_scores, perans_scores = [], [], []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                    image_dir_path=args.ok_vqa_image_dir_path,
                    questions_json_path=args.ok_vqa_questions_json_path,
                    annotations_json_path=args.ok_vqa_annotations_json_path,
                    additional_shots=args.additional_shots,
                    max_generation_length=args.max_generation_length,
                    min_generation_length=args.min_generation_length,
                    vqa_dataset="ok_vqa",
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    conv_mode=args.conv_mode,
                )
                if type(ok_vqa_score) == dict:
                    output_score_detials = True
                    overall_score = ok_vqa_score['overall']
                    perques_score = ok_vqa_score['perQuestionType']
                    perans_score = ok_vqa_score['perAnswerType']
                    perques_scores.append(perques_score)
                    perans_scores.append(perans_score)
                    ok_vqa_score = overall_score
                print(f"Shots {shot} Trial {trial} VQA score: {ok_vqa_score}")
                scores.append(ok_vqa_score)

            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            if output_score_detials:
                perques_score_mean = {key: np.mean([item[key]]) for key in perques_scores[0].keys() for item in
                                      perques_scores}
                perans_score_mean = {key: np.mean([item[key]]) for key in perans_scores[0].keys() for item in
                                     perans_scores}
                perques_score_std = {key: np.std([item[key]]) for key in perques_scores[0].keys() for item in
                                     perques_scores}
                perans_score_std = {key: np.std([item[key]]) for key in perans_scores[0].keys() for item in
                                    perans_scores}
                results["ok_vqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.mean(scores),
                     "std": np.std(scores), "perQuestion_mean": perques_score_mean,
                     "perQuestion_std": perques_score_std, "perAnswer_mean": perans_score_mean,
                     "perAnswer_std": perans_score_std}
                )
            results["ok_vqa"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            output_score_detials = False
            scores, perques_scores, perans_scores = [], [], []
            if args.vqav2_img_mode != "plain":
                severities = args.vqav2_img_noise_severity
                noisy_flag = True
            else:
                severities = [0]
                noisy_flag = False
            for severity in severities:
                image_dir_path = f"mscoco/val2014-c/{args.vqav2_img_mode}/{severity}/val2014" \
                    if noisy_flag else args.vqav2_image_dir_path
                vqa_score = evaluate_vqa(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=args.trial_seeds[0],
                    image_dir_path=image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    max_generation_length=args.max_generation_length,
                    min_generation_length=args.min_generation_length,
                    additional_shots=args.additional_shots,
                    vqa_dataset="vqa",
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    conv_mode=args.conv_mode,
                )
                if type(vqa_score) == dict:
                    output_score_detials = True
                    overall_score = vqa_score['overall']
                    perques_score = vqa_score['perQuestionType']
                    perans_score = vqa_score['perAnswerType']
                    perques_scores.append(perques_score)
                    perans_scores.append(perans_score)
                    vqa_score = overall_score
                print(f"Shots {shot} VQA score: {vqa_score}")
                scores.append(vqa_score)

            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            if output_score_detials:
                perques_score_mean = {key: np.mean([item[key]]) for key in perques_scores[0].keys() for item in
                                      perques_scores}
                perans_score_mean = {key: np.mean([item[key]]) for key in perans_scores[0].keys() for item in
                                     perans_scores}
                perques_score_std = {key: np.std([item[key]]) for key in perques_scores[0].keys() for item in
                                     perques_scores}
                perans_score_std = {key: np.std([item[key]]) for key in perans_scores[0].keys() for item in
                                    perans_scores}
                results["vqav2"].append(
                    {"shots": shot, "trials": scores, "mean": np.mean(scores),
                     "std": np.std(scores), "perQuestion_mean": perques_score_mean,
                     "perQuestion_std": perques_score_std, "perAnswer_mean": perans_score_mean,
                     "perAnswer_std": perans_score_std}
                )
            else:
                results["vqav2"].append(
                    {"shots": shot, "trials": scores, "mean": np.mean(scores), "std": np.std(scores)}
                )

    if args.eval_pope:
        for questions_path in ["coco_pope_popular.json", "coco_pope_random.json", "coco_pope_adversarial.json"]:
            print(f"Evaluating {questions_path} task in POPE..")
            questions_json_path = os.path.join(args.pope_annot_dir, questions_path)
            scores = evaluate_pope(
                eval_model,
                batch_size=args.batch_size,
                image_dir_path=os.path.join(DATA_DIR, args.pope_image_dir_path),
                questions_json_path=questions_json_path,
                seed=args.trial_seeds[0],
                max_generation_length=args.max_generation_length,
                min_generation_length=args.min_generation_length,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                conv_mode=args.conv_mode,
            )
            results[questions_path] = [{
                "acc": scores[0],
                "precision": scores[1],
                "recall": scores[2],
                "f1": scores[3],
                "yes_ratio": scores[4],
            }]

    if args.eval_mme:
        mme_test_path=None
        for task_type in mme_eval_type_dict:
            for task in mme_eval_type_dict[task_type]:
                print(f"Evaluating {task} of {task_type} in MME benchmark..")
                mme_test_path = evaluate_mme(
                    eval_model,
                    batch_size=args.batch_size,
                    eval_folder=os.path.join(args.mme_annot_dir, task),
                    seed=args.trial_seeds[0],
                    max_generation_length=args.max_generation_length,
                    min_generation_length=args.min_generation_length,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    task_name=task,
                    task_type=task_type,
                    model_name=ckpt_model+"-"+peft_model,
                    conv_mode=args.conv_mode,
                )
        assert mme_test_path is not None
        mme_results_dict = get_final_mme_scores(mme_test_path)
        results["mme"] = [mme_results_dict]

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
) -> list:
    context_text = []
    if effective_num_shots > 0:
        for s in in_context_samples:
            context_text.append(get_prompt(s))
    return context_text


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def evaluate_coco_flickr(
    eval_model,
    batch_size,
    image_dir_path,
    annotations_json_path,
    demo_annotations_json_path,
    seed=42,
    max_generation_length=20,
    min_generation_length=1,
    num_beams=5,
    length_penalty=1.0,
    num_samples=5000,
    query_set_size=5000,
    is_flickr=False,
    additional_shots=2,
    do_sample=False,
    top_k=5,
    eval_blip=False,
    conv_mode="v0",

):
    """Evaluate a model on COCO dataset.
    """
    eval_dataset = COCOFlickrDataset(
        image_dir_path=os.path.join(DATA_DIR, image_dir_path),
        annotations_path=os.path.join(DATA_DIR, annotations_json_path),
        is_flickr=is_flickr,
    )
    demo_dataset = COCOFlickrCaptionDataset(
        annotations_path=os.path.join(DATA_DIR, demo_annotations_json_path),
        is_flickr=is_flickr,
    )
    ## effective_num_shots is for text instruction only
    num_shots=0
    if additional_shots == 0:
        effective_num_shots = num_shots
    else:
        effective_num_shots = num_shots if num_shots > 0 else additional_shots

    if effective_num_shots > len(demo_dataset):
        raise ValueError(
            f"num_shots must be less than or equal to {len(demo_dataset)}"
        )
    
    def get_prompt(sample, train=True):
        caption = np.random.choice(sample['captions'])
        if conv_mode=="llava_llama_2":
            if train: ## prompts for few-shot testing
                return_list = [
                    "Please provide a description of the picture.",
                    caption.strip()
                ]
            else:
                return_list = [
                    "Please provide a description of the picture."
                ]
        elif conv_mode == "v0":
            if train: ## prompts for few-shot testing
                return_list = [
                    "Please provide a description of the picture.",
                    caption.strip()
                ]
            else:
                return_list = return_list = [
                    "Please provide a description of the picture."
                ]
        else:
            raise NotImplementedError
        return return_list

    random_indices = get_random_indices(0, query_set_size, demo_dataset, seed)

    in_context_samples, _ = prepare_eval_samples_and_dataset(
        full_dataset=demo_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
    )

    predictions = defaultdict()

    desc = "Running inference Flickr30" if is_flickr else "Running inference COCO"

    for batch in more_itertools.chunked(tqdm(eval_dataset, desc=desc), batch_size):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_images = []
        batch_text = []
        batch_prompt = []
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])

            context_text = get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
            )

            batch_text.append(get_prompt(batch[i], train=False, )[0])
            batch_prompt.append(context_text)

        outputs  = eval_model.get_outputs(
            batch_images=batch_images,
            prompt_texts=batch_prompt,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            min_generation_length=min_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            top_k=top_k,

        )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_name"]] = {
                "caption": new_predictions[i],
            }

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    test_results_dir = "test_results" if not eval_blip else "test_results_instblip"
    os.makedirs(test_results_dir, exist_ok=True)
    results_path = (
        f"flickrresults_{random_uuid}.json"
        if is_flickr
        else f"cocoresults_{random_uuid}.json"
    )
    with open(f"./{test_results_dir}/{results_path}", "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_name": k, "caption": predictions[k]["caption"]}
                    for k in predictions
                ],
                indent=4,
            )
        )

    metrics = compute_caption_metrics(
        result_path=f"./{test_results_dir}/{results_path}",
        annotations_path=os.path.join(DATA_DIR, annotations_json_path),
    )

    # delete the temporary file
    os.remove(f"./{test_results_dir}/{results_path}")
    metrics = {item: metrics[item] * 100.0 for item in metrics.keys()}

    return metrics

def evaluate_vqa(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path,
    annotations_json_path,
    seed=42,
    max_generation_length=10,
    min_generation_length=1,
    num_beams=5,
    length_penalty=-1.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    additional_shots=2,
    do_sample=False,
    top_k=5,
    vqa_dataset="vqa",
    eval_blip=False,
    conv_mode="v0",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.
    """
    setup_seed(seed)
    full_dataset = VQADataset(
        image_dir_path=os.path.join(DATA_DIR, image_dir_path),
        question_path=os.path.join(DATA_DIR, questions_json_path),
        annotations_path=os.path.join(DATA_DIR, annotations_json_path),
        vqa_dataset=vqa_dataset,
    )

    ## effective_num_shots is for text instruction only
    if additional_shots == 0:
        effective_num_shots = num_shots
    else:
        effective_num_shots = num_shots if num_shots > 0 else additional_shots

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)

    def get_prompt(sample, train=True,):
        if train:
            return_list = [
                sample['question'].strip(),
                sample['answers'][0].strip()
            ]
        else:
            return_list = [
                sample['question'].strip()
            ]
        return return_list

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
    )

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_images = []
        batch_text = []
        batch_prompt = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            ## no context image for mplug-owl (1 image per instance)
            batch_images.append([batch[i]["image"]])

            context_text = get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
            )

            batch_text.append(get_prompt(batch[i], train=False, )[0])
            batch_prompt.append(context_text)

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            prompt_texts=batch_prompt,
            max_generation_length=max_generation_length,
            min_generation_length=min_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            top_k=top_k,
        )

        process_function = (
            postprocess_vqa_generation
            if vqa_dataset == "vqa"
            else postprocess_ok_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    test_results_dir = "test_results" if not eval_blip else "test_results_instblip"
    os.makedirs(test_results_dir, exist_ok=True)
    with open(f"./{test_results_dir}/{vqa_dataset}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"./{test_results_dir}/{vqa_dataset}results_{random_uuid}.json",
        os.path.join(DATA_DIR, questions_json_path),
        os.path.join(DATA_DIR, annotations_json_path),
    )

    # delete the temporary file
    os.remove(f"./{test_results_dir}/{vqa_dataset}results_{random_uuid}.json")

    return acc

def evaluate_pope(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path,
    seed=42,
    max_generation_length=10,
    min_generation_length=1,
    num_beams=5,
    length_penalty=-1.0,
    do_sample=False,
    top_k=5,
    conv_mode="v0",
):
    """
    Evaluate a model on POPE benchmark.
    """
    setup_seed(seed)
    full_dataset = PopeDataset(
        image_dir_path=os.path.join(DATA_DIR, image_dir_path),
        question_path=questions_json_path,
    )

    def get_prompt(sample, train=True,):
        if train:
            return_list = [
                sample['question'].strip(),
                sample['answers'][0].strip()
            ]
        else:
            return_list = [
                sample['question'].strip()
            ]
        return return_list

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(full_dataset, desc="Running inference"), batch_size
    ):

        batch_images = []
        batch_text = []

        ## get example batch
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(get_prompt(batch[i], train=False, )[0])

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            prompt_texts = [[]], ## zero-shot for pope evaluation
            max_generation_length=max_generation_length,
            min_generation_length=min_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            top_k=top_k,
        )


        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"], "label": sample["label"]}
                for p, sample in zip(outputs, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)
    out_file_path = f"./{test_results_dir}/pope_results_{random_uuid}.json"
    with open(out_file_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    _, metrics = get_pope_results(out_file_path)

    # delete the temporary file
    os.remove(out_file_path)

    return metrics

def evaluate_mme(
    eval_model,
    batch_size,
    eval_folder,
    seed=42,
    max_generation_length=10,
    min_generation_length=1,
    num_beams=5,
    length_penalty=-1.0,
    do_sample=False,
    top_k=5,
    task_name="",
    task_type="",
    model_name="",
    conv_mode="v0",
):
    """
    Evaluate a model on MME benchmark.
    """
    setup_seed(seed)
    full_dataset = MmeDataset(
        eval_folder=eval_folder,
        task_type=task_type,
    )

    mme_write_in_format = "{}\t{}\t{}\t{}"

    def get_prompt(sample, train=True,):
        if train:
            return_list = [
                sample['question'].strip(),
                sample['answers'][0].strip()
            ]
        else:
            return_list = [
                sample['question'].strip()
            ]
        return return_list

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(full_dataset, desc="Running inference"), batch_size
    ):

        batch_images = []
        batch_text = []

        ## get example batch
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(get_prompt(batch[i], train=False, )[0])

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            prompt_texts=[[]],  ## zero-shot for pope evaluation
            max_generation_length=max_generation_length,
            min_generation_length=min_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            top_k=top_k,
        )


        predictions.extend(
            [
                mme_write_in_format.format(sample['image_name'], sample['question'], sample['label'], p.replace("\t", ""))
                for p, sample in zip(outputs, batch)
            ]
        )
    # save the predictions to a temporary file
    test_results_dir = f"./test_results/mme_results/{model_name}"
    os.makedirs(test_results_dir, exist_ok=True)
    out_file_path = f"{test_results_dir}/{task_name}.txt"
    with open(out_file_path, "w") as f:
        for line in predictions:
            line = line.replace("\n", "")
            f.write(line.strip() + "\n")

    return test_results_dir


if __name__ == "__main__":
    main()
