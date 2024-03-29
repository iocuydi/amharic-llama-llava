# Amharic LLaVA


# Inference
1. Download llama2 weights and convert to huggingface as shown here https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py 
2. Download amharic llava weights and the PRETRAINED amharic llama weights
3. Download this clip model: https://huggingface.co/openai/clip-vit-large-patch14-336
4. Edit llava/eval/llava_inf.sh with the appropriate paths
5. Run llava/eval/llava_inf.sh, when promtped input image path and text prompt



# Training
1. Set up datasets. We use the LLaVA 1.5 procedure with pretraining to align clip with an adapter, followed by finetuning. The datasets used are the same ones use the same images as here (https://github.com/haotian-liu/LLaVA#pretrain-feature-alignment) minus a few that had poor translation. Set up your image folders the same way as LLaVA describes, then switch out the JSON files with ours from huggingface (blip laion for pretraining, amharic visual instruction tuning for finetuning).
2. Use scripts/v1_5/pretrain.sh to pretrain an adapter. 
3. Use scripts/v1_5/finetune_lora.sh to finetune based on the adapter. (We did not finetune without lora, but should be possible)
4. Make sure to validate that the jsons are pointing to the right images prior to training.

Note that the adapter is merged and saved alongside the other weights after finetuning.

# Custom Datasets
We did not try this, but it should work. Follow the instructions from LLaVA https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md


