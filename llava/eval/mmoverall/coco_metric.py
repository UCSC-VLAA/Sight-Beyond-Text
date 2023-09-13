# from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import json

class COCOEvalCap:
    def __init__(self, path_pred, path_labels):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        labels = json.load(open(path_labels))
        preds = json.load(open(path_pred))
        imgids = [item['image_name'] for item in preds]
        self.ground_truth, self.prediction = {}, {}
        for label_item in labels:
            self.ground_truth[label_item['image_name']] = [{'caption': i} for i in label_item['captions']]
        for pred_item in preds:
            self.prediction[pred_item['image_name']] = [{'caption': pred_item['caption']}]
        self.params = {'image_id': imgids}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.ground_truth[imgId]
            res[imgId] = self.prediction[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
# def compute_cider(
#     result_path,
#     annotations_path="/data/yfcc-tmp/data/mscoco/annotations/captions_train2017.json",
# ):
#     # create coco object and coco_result object
#     coco = COCO(annotations_path)
#     coco_result = coco.loadRes(result_path)
#
#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)
#     coco_eval.params["image_id"] = coco_result.getImgIds()
#     coco_eval.evaluate()
#
#     return coco_eval.eval

def compute_caption_metrics(
    result_path,
    annotations_path="/data/yfcc-tmp/data/mscoco/annotations/captions_train2017.json",
):
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(result_path, annotations_path)
    coco_eval.evaluate()
    output = {}
    for metric, score in coco_eval.eval.items():
        output[metric] = score
    return output

def postprocess_captioning_generation(predictions):
    return predictions
