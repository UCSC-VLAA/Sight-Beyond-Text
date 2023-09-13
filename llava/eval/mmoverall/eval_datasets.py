import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class COCOFlickrDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if "image_id" in self.annotations[idx].keys():
            img_pth = os.path.join(
                self.image_dir_path, f"COCO_test2014_{self.annotations[idx]['image_id']:012d}.jpg"
            )
        else:
            img_pth = os.path.join(
                self.image_dir_path, self.annotations[idx]["image_name"]
            )
        return img_pth

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx)).convert('RGB')
        image.load()
        captions = self.annotations[idx]["captions"]
        return {
            "image": image,
            "captions": captions,
            "image_name": self.annotations[idx]["image_name"],
        }

class COCOFlickrCaptionDataset(Dataset):
    def __init__(
        self,
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        is_flickr=False,
    ):
        self.annotations = json.load(open(annotations_path))
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        captions = self.annotations[idx]["captions"]
        return {
            "captions": captions,
            "image_name": self.annotations[idx]["image_name"],
        }


class VQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
        question_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
        vqa_dataset="vqa",
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.vqa_dataset = vqa_dataset

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.vqa_dataset == "vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
            )
        elif self.vqa_dataset == "ok_vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
            )
        else:
            raise Exception(f"Unknown VQA dataset {self.vqa_dataset}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path).convert('RGB')
        image.load()
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }

class MmeDataset(Dataset):
    """Class to represent the ImageNet1k dataset."""

    def __init__(
            self,
            eval_folder="./mme/text_translation",
            task_type="Cognition",
    ):
        assert task_type in ["Cognition", "Perception"]
        files = os.listdir(eval_folder)
        self.examples = []
        if len(files) == 2:
            # assert files == ["questions_answers_YN", "images"]
            for file in os.listdir(os.path.join(eval_folder, "questions_answers_YN")):
                question_id, file_type = file.split(".")
                with open(os.path.join(eval_folder, "questions_answers_YN", file), 'r') as f:
                    texts = f.readlines()
                texts = [item.strip() for item in texts]
                for text in texts:
                    text, label = text.split("\t")
                    image_prefix = os.path.join(eval_folder, "images", question_id)
                    self.examples.append(
                        {
                            "text": text,
                            "label": label,
                            "image": image_prefix + ".jpg" if task_type == "Perception" else image_prefix + ".png"
                        }
                    )
        else:
            for file in files:
                question_id, file_type = file.split(".")
                if file_type == "txt":
                    with open(os.path.join(eval_folder, file), 'r') as f:
                        texts = f.readlines()
                    texts = [item.strip() for item in texts]
                    for text in texts:
                        image_prefix = os.path.join(eval_folder, question_id)
                        text, label = text.split("\t")
                        self.examples.append(
                            {
                                "text": text,
                                "label": label,
                                "image": image_prefix + ".jpg" if task_type == "Perception" else image_prefix + ".png"
                            }
                        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        cur_example = self.examples[idx]
        img_path = cur_example['image']
        image = Image.open(img_path).convert('RGB')
        image.load()
        return {
            "image_name": img_path.split("/")[-1],
            "image": image,
            "question": cur_example['text'],
            "label": cur_example['label'],
        }


class PopeDataset(Dataset):
    """Class to represent the ImageNet1k dataset."""

    def __init__(
            self,
            image_dir_path="mscoco/val2014",
            question_path="pope/output/coco/coco_pope_adversarial.json",
    ):
        self.question_annot = [json.loads(q) for q in open(question_path, 'r')]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.question_annot)

    def get_img_path(self, instance):
        return os.path.join(
            self.image_dir_path, instance
        )

    def __getitem__(self, idx):
        question = self.question_annot[idx]['text']
        label = self.question_annot[idx]['label']
        question_id = self.question_annot[idx]['question_id']
        img_path = self.get_img_path(self.question_annot[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        image.load()
        return {
            "image": image,
            "question": question,
            "label": label,
            "question_id": question_id,
        }
