This repo and evaluate specific on quantization in flickr30k
Add --save_model to save models to pth for easy loading

python run_with_submitit.py --backbone "timm_tf_efficientnet_b3_ns" --dataset_config configs/flickr.json --resume weights/pretrained_EB3_checkpoint.pth  --ngpus 1 --nodes 1 --eval --quantization fp16

python -m torch.distributed.launch --use_env main.py --dataset_config configs/flickr.json --load weights/pretrained_EB3_checkpoint.pth --backbone timm_tf_efficientnet_b3_ns --text_encoder_lr 5e-5 --batch_size 1 --epochs 5 --epoch_chunks 25 --output-dir checkpoint/ --finetune_LLM --replace_LLM

python  main.py --dataset_config configs/flickr.json --load weights/pretrained_EB3_checkpoint.pth --backbone timm_tf_efficientnet_b3_ns --text_encoder_lr 5e-5 --batch_size 2 --epochs 5 --epoch_chunks 25 --output-dir checkpoint/ --finetune_LLM --replace_LLM > checkpoint/anhnguyen/experiments/training.log