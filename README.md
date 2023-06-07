# VanillaNet: the Power of Minimalism in Deep Learning 
<p align="left">
<a href="https://arxiv.org/abs/2305.12972" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2305.12972-b31b1b.svg?style=flat" /></a>
</p>


Official PyTorch implementation of **VanillaNet**, from the following paper:\
[VanillaNet: the Power of Minimalism in Deep Learning ](https://arxiv.org/abs/2305.12972)\
Hanting chen, [Yunhe Wang](https://www.wangyunhe.site/), Jianyuan Guo and Dacheng Tao

<img src="pic/structure.PNG" width="800px"/>

VanillaNet is an innovative neural network architecture that focuses on **simplicity** and **efficiency**. Moving away from complex features such as **shortcuts** and **attention** mechanisms, VanillaNet uses a reduced number of layers while still **maintaining excellent performance**. This project showcases that it's possible to achieve effective results with a lean architecture, thereby setting a new path in the field of computer vision and challenging the status quo of foundation models. 

## News

**2023.06.02** In addition to the reported speed in the paper, we have also measured the speed with NVIDIA TensorRT on A100 and the speed on HUAWEI Ascend 910. The inference speed of VanillaNet is superior to other counterparts. 🍺


## Comparison of Depth and Speed

<img src="pic/depth.PNG" width="360px"/> <img src="pic/speed.PNG" width="300px"/>

VanillaNet achieves comparable performance to prevalent computer vision foundation models, yet boasts a **reduced depth and enhanced inference speed**:
- **11-layers'** VanillaNet achieves about **81%** Top-1 accuracy with **3.59ms**, over **100%** speed increase compared to ResNet-50 (**7.64ms**).
- **13 layers'** VanillaNet (1.5x*) achieves about **83%** Top-1 accuracy with **9.72ms**, over **100%** speed increase compared to Swin-S (**20.25ms**).
- With tensorRT FP32 on A100, **11 layers'** VanillaNet achieves about **81%** Top-1 accuracy with **0.69ms**, over **100%** speed increase compared to Swin-T (**1.41ms**) and ResNet-101 (**1.58ms**).

| name | Params(M) | FLOPs(B) | Latency(ms) <br/>Pytorch <br/>A100 | Latency(ms) <br/>MindSpore <br/>Ascend 910 | Latency(ms) <br/>TRT FP32 <br/>A100 | Latency(ms) <br/>TRT FP16 <br/>A100 | Acc(%) |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| Swin-T | 28.3 | 4.5 | 10.51 | 2.24 | 1.41 | 0.98 | 81.18 |
| ResNet-18 | 11.7 | 1.8 | 3.12 | 0.60 | 0.41 | 0.28 | 70.6 |
| ResNet-34 |21.8|3.7|5.57|0.97|0.77|0.49|75.5|
| ResNet-50 |25.6|4.1|7.64|1.23|0.80|0.54|79.8|
| ResNet-101 |45.0|8.0|-|2.34|1.58|1.04|81.3|
| ResNet-152 |60.2|11.5|-|3.40|2.30|1.49|81.8|
| **VanillaNet-5** | 15.5 | 5.2 | 1.61 |0.47|0.33|0.27| 72.49 |
| **VanillaNet-6** | 32.5 | 6.0 | 2.01 |0.61|0.40|0.33| 76.36 |
| **VanillaNet-7** | 32.8 | 6.9 | 2.27 | 0.88 |0.47|0.39|77.98 |
| **VanillaNet-8** | 37.1 | 7.7 | 2.56 |0.96|0.52|0.45| 79.13 |
| **VanillaNet-9** | 41.4 | 8.6 | 2.91 |1.02|0.58|0.49| 79.87 |
| **VanillaNet-10** | 45.7 | 9.4 | 3.24 |1.11|0.63|0.53| 80.57 |
| **VanillaNet-11** | 50.0 | 10.3 | 3.59 | 1.17 |0.69|0.58| 81.08 |
| **VanillaNet-12** | 54.3 | 11.1 | 3.82 |1.26|0.75|0.62| 81.55 |
| **VanillaNet-13** | 58.6 | 11.9 | 4.26 |1.33|0.82|0.67| 82.05 |

## Downstream Tasks

Please refer to [this page](https://github.com/huawei-noah/VanillaNet/tree/main/object_detection).

VanillaNet achieves a higher Frames Per Second (FPS) in **detection** and **segmentation** tasks.



## Catalog
- [x] ImageNet-1K Testing Code  
- [x] ImageNet-1K Training Code of VanillaNet-5 to VanillaNet-10  
- [x] ImageNet-1K Pretrained Weights of VanillaNet-5 to VanillaNet-10
- [ ] ImageNet-1K Training Code of VanillaNet-11 to VanillaNet-13
- [x] ImageNet-1K Pretrained Weights of VanillaNet-11 to VanillaNet-13
- [x] Downstream Transfer (Detection, Segmentation) Code

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | #params(M) | FLOPs(B) | Lacency(ms) | Acc(%) | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| VanillaNet-5 | 15.5 | 5.2 | 1.61 | 72.49 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_5.pth) |
| VanillaNet-6 | 32.5 | 6.0 | 2.01 | 76.36 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_6.pth) |
| VanillaNet-7 | 32.8 | 6.9 | 2.27 | 77.98 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_7.pth) |
| VanillaNet-8 | 37.1 | 7.7 | 2.56 | 79.13 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_8.pth) |
| VanillaNet-9 | 41.4 | 8.6 | 2.91 | 79.87 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_9.pth) |
| VanillaNet-10 | 45.7 | 9.4 | 3.24 | 80.57 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_10.pth) |
| VanillaNet-11 | 50.0 | 10.3 | 3.59 | 81.08 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_11.pth) |
| VanillaNet-12 | 54.3 | 11.1 | 3.82 | 81.55 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_12.pth) |
| VanillaNet-13 | 58.6 | 11.9 | 4.26 | 82.05 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_13.pth) |
| VanillaNet-13-1.5x | 127.8 | 26.5 | 7.83 | 82.53 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_13_x1_5.pth) |
| VanillaNet-13-1.5x&dagger; | 127.8 | 48.9 | 9.72 | 83.11 | [model](https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_13_x1_5_ada_pool.pth) |

## Installation

The results are produced with `torch==1.10.2+cu113 torchvision==0.11.3+cu113 timm==0.6.12`. Other versions might also work.

Install [Pytorch](https://pytorch.org/) and, [torchvision](https://pytorch.org/vision/stable/index.html) following official instructions.

Install required packages:
```
pip install timm==0.6.12
pip install cupy-cuda113
pip install torchprofile
pip install einops
pip install tensorboardX
pip install terminaltables
```

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```


## Testing

We give an example evaluation command  for VanillaNet-5:

without deploy:

```
python -m torch.distributed.launch --nproc_per_node=1 main.py --model vanillanet_5 --data_path /path/to/imagenet-1k/ --real_labels /path/to/imagenet_real_labels.json --finetune /path/to/vanillanet_5.pth --eval True --model_key model_ema --crop_pct 0.875
```

with deploy:
```
python -m torch.distributed.launch --nproc_per_node=1 main.py --model vanillanet_5 --data_path /path/to/imagenet-1k/ --real_labels /path/to/imagenet_real_labels.json --finetune /path/to/vanillanet_5.pth --eval True --model_key model_ema --crop_pct 0.875 --switch_to_deploy /path/to/vanillanet_5_deploy.pth
```

## Training

You can use the following command to train VanillaNet-5 on a single machine with 8 GPUs: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_5 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 3.5e-3 --weight_decay 0.35  --drop 0.05 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.1 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*1 = 1024`.

To train other VanillaNet variants, `--model` need to be changed. Examples are given below.

<details>
<summary>
VanillaNet-6
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_6 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 4.8e-3 --weight_decay 0.32  --drop 0.05 \
--layer_decay 0.8 --layer_decay_num_layers 4 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.15 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```
</details>

<details>
<summary>
VanillaNet-7
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_7 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 4.7e-3 --weight_decay 0.35  --drop 0.05 \
--layer_decay 0.8 --layer_decay_num_layers 5 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.4 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```
</details>

<details>
<summary>
VanillaNet-8
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_8 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 3.5e-3 --weight_decay 0.3  --drop 0.05 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.4 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```
</details>

<details>
<summary>
VanillaNet-9
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_9 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 3.5e-3 --weight_decay 0.3  --drop 0.05 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.4 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```
</details>

<details>
<summary>
VanillaNet-10
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vanillanet_10 \
--data_path /path/to/imagenet-1k \
--batch_size 128 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 3.5e-3 --weight_decay 0.25  --drop 0.05 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.4 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true 
```
</details>


### Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [RegVGG](https://github.com/DingXiaoH/RepVGG), and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repositories.

### License
This project is released under the MIT license. Please see the [LICENSE](License) file for more information.

### Instruction pdf

A instruction pdf (Chinese version) can be found [here](https://github.com/huawei-noah/VanillaNet/blob/main/pic/vanillanet.pdf)

### Citation
If our work is useful for your research, please consider citing:
```
@article{chen2023vanillanet,
  title={VanillaNet: the Power of Minimalism in Deep Learning},
  author={Chen, Hanting and Wang, Yunhe and Guo, Jianyuan and Tao, Dacheng},
  journal={arXiv preprint arXiv:2305.12972},
  year={2023}
}
```
