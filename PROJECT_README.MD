This repo and evaluate specific on quantization in flickr30k
Add --save_model to save models to pth for easy loading

python run_with_submitit.py --backbone "timm_tf_efficientnet_b3_ns" --dataset_config configs/flickr.json --resume weights/pretrained_EB3_checkpoint.pth  --ngpus 1 --nodes 1 --eval --quantization fp16