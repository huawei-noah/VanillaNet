# COCO Object detection with VanillaNet

## Getting started 

We add VanillaNet model and config files based on [mmdetection-2.x](https://github.com/open-mmlab/mmdetection/tree/2.x). Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/get_started.md) for mmdetection installation and dataset preparation instructions.

## Results and Fine-tuned Models

| Framework | Backbone | FLOPs(G) | Params(M) | FPS | AP<sup>b</sup> | AP<sup>m</sup> | Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| RetinaNet | Swin-T | 244.8 | 38.5 | 27.5 | 41.5 | - |-|
|  | VanillaNet-13 | 396.9 | 75.4 | 29.8 | 43.0 | - | [log](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/retinanet_vanillanet_13.log.json)/[model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/retinanet_vanillanet_13.pth) |
| Mask RCNN | [Swin-T](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/master) | 263.8 | 47.8 | 28.2 | 43.7 | 39.8 |-|
|  | ConvNeXtV2-Nano | 220.6 | 35.2 | 34.4 | 43.3 | 39.4 |-|
|  | VanillaNet-13 | 420.7 | 77.1 | 32.6 | 44.3 | 40.1 | [log](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/mask_rcnn_vanillanet_13.log.json)/[model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/mask_rcnn_vanillanet_13.pth) |


### Training

You can download the ImageNet pre-trained [checkpoint](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_13_act_num_4_kd_pretrain.pth) for VanillaNet-13(act_num=4), which is trained via [knowledge distillation(this paper)](https://arxiv.org/pdf/2305.15781.pdf).

For example, to train a Mask R-CNN model with VanillaNet backbone and 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/vanillanet/mask_rcnn_vanillanet_13_mstrain_480-1024_adamw.py --gpus 8 --launcher pytorch --work-dir <WORK_DIR>
```

### Inference

For example, test with single-gpu, run:
```
python -m torch.distributed.launch --nproc_per_node=1 tools/test.py configs/vanillanet/mask_rcnn_vanillanet_13_mstrain_480-1024_adamw.py <CHECKPOINT_FILE> --launcher pytorch --eval bbox segm
```

## Acknowledgment 

This code is built based on [mmdetection](https://github.com/open-mmlab/mmdetection), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repositories.