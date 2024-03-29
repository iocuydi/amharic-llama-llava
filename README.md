# amharic-llama-llava

Pretraining, finetuning, and inference for Amharic LLaMA and LLaVA adapted from:

https://github.com/ymcui/Chinese-LLaMA-Alpaca

https://github.com/facebookresearch/llama-recipes

https://github.com/haotian-liu/LLaVA

Llama-2-Amharic weights: https://huggingface.co/iocuydi/llama-2-amharic-3784m
Can be run with the inference script in this repo. Pretrained on 3.784b Amharic tokens.

Amharic LLaVA weights: https://huggingface.co/iocuydi/amharic-llava
Can be run with the inference script in this repo. Must run with the PRETRAINED amharic llama, not finetuned.
See llava eval section for more details.

Amharic LLaVA requires this CLIP model: https://huggingface.co/openai/clip-vit-large-patch14-336

Associated datasets:

Amharic Blip Laion: https://huggingface.co/datasets/iocuydi/amharic-blip-laion
Amharic Dolly: https://huggingface.co/datasets/iocuydi/amharic-dolly-15k
Amharic Alpaca: https://huggingface.co/datasets/iocuydi/amharic-alpaca
Amharic Visual Instruction Tuning: https://huggingface.co/datasets/iocuydi/amharic-visual-instruction-tuning
Amharic RedPajama Synthetic (pretraining, partial): https://huggingface.co/datasets/iocuydi/amharic-redpajama-synthetic
Amharic OASST1 Pruned: https://huggingface.co/datasets/iocuydi/amharic-OASST1-pruned

More info
https://arxiv.org/abs/2403.06354
https://medium.com/@garrilogistics/llama-2-amharic-llms-for-low-resource-languages-d6fb0ba332f4

Cite: 
```
@misc{andersland2024amharic,
      title={Amharic LLaMA and LLaVA: Multimodal LLMs for Low Resource Languages}, 
      author={Michael Andersland},
      year={2024},
      eprint={2403.06354},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
