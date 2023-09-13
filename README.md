# Sight Beyond Text: Multi-Modal Training Enhances LLMs in Truthfulness and Ethics



[Haoqin Tu*](https://www.haqtu.me/), [Bingchen Zhao*](https://bzhao.me), [Chen Wei](https://weichen582.github.io/), [Cihang Xie](https://cihangxie.github.io/) (*Equal Contribution)



[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

## Train on Text-only Instructions

To train the model on text-only instruction data, first you need to re-organize the data by removing all `<image>` placeholders in the data:

```python
import json
llava_80k_data = json.load(open("path/to/llava_instruct_80k.json"))
llava_80k_text_data = []
for data in llava_80k_data:
    data['conversations'][0]['value'] = data['conversations'][0]['value'].replace("<image>", "")
    llava_80k_text_data.append(data)

with open("path/to/llava_text_instruct_80k.json", 'w') as f:
    json.dump(llava_80k_text_data, f, indent=4)
```

Then run `train_scripts/finetune-lm-7b.sh` or `train_scripts/lora-lm-7b.sh` with your specified data path.

## Evaluations

For NLP & Multi-Modal data and evaluations, please see instructions [here](./llava/eval/README.md).

**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.



## Citation

If you find this repo useful for your your research and applications, please cite using this BibTeX:
```bibtex
@article{zhao2023sight,
  title={Sight Beyond Text: Open the Eyes of LLMs Unlocks Emerging Benefits in Truthfulness and Ethics},
  author={Tu, Haoqin and Zhao, Bingchen and Wei, Chen and Xie, Cihang},
  journal={arXiv preprint arXiv:2308.xxxxx},
  year={2023}
}
```

## Acknowledgement

This work is partially supported by a gift from Open Philanthropy. We thank Center for AI Safety for supporting our computing needs. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the sponsors.

## Related Projects

TODO



