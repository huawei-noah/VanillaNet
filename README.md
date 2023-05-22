# VanillaNet: the Power of Minimalism in Deep Learning 


Official PyTorch implementation of **VanillaNet**, from the following paper:\
**VanillaNet: the Power of Minimalism in Deep Learning**\
*Hanting chen, [Yunhe Wang](https://www.wangyunhe.site/), Jianyuan Guo and Dacheng Tao*

<img src="pic/structure.PNG" width="800px"/>

VanillaNet is an innovative neural network architecture that focuses on **simplicity** and **efficiency**. Moving away from complex features such as **shortcuts** and **attention** mechanisms, VanillaNet uses a reduced number of layers while still **maintaining excellent performance**. This project showcases that it's possible to achieve effective results with a lean architecture, thereby setting a new path in the field of computer vision and challenging the status quo of foundation models. 

## Comparison of Depth and Speed

<img src="pic/depth.PNG" width="360px"/> <img src="pic/speed.PNG" width="300px"/>

VanillaNet, in its robust simplicity, offers comparable precision to prevalent computer vision foundation models, yet boasts a **reduced depth and enhanced processing speed** （test on Nvidia A100 GPU with batch size 1):
- **9-layers'** VanillaNet achieves about **80%** Top-1 accuracy with **3.59ms**, over **100%** speed increase compared to ResNet-50 (**7.64ms**).
- **13 layers'** VanillaNet achieves about **83%** Top-1 accuracy with **9.72ms**, over **100%** speed increase compared to Swin-T (**20.25ms**).

## Downstream Tasks

| Framework | Backbone | FLOPs(G) | #params(M) | FPS | AP<sup>b</sup> | AP<sup>m</sup> |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| RetinaNet | Swin-T| 245 | 38.5 | 27.5 | 41.5 | - |
|  | VanillaNet-11 | 386 | 67.0 | 30.8 | 41.8 | - |
| Mask RCNN | ConvNeXtV2-N | 221 | 35.2 | 31.7 | 42.7 | 38.9 |
|  | [Swin-T](https://github.com/open-mmlab/mmdetection/tree/main/configs/swin) | 267 | 47.8 | 28.2 | 42.7 | 39.3 |
|  | VanillaNet-11 | 404 | 107.5 | 33.6 | 42.9 | 39.6 |

VanillaNet achieves a higher Frames Per Second (FPS) in **detection** and **segmentation** tasks.



## Catalog
- [ ] ImageNet-1K Testing Code  
- [ ] ImageNet-1K Training Code of VanillaNet-5 to VanillaNet-10  
- [ ] ImageNet-1K Pretrained Weights of VanillaNet-5 to VanillaNet-10
- [ ] ImageNet-1K Training Code of VanillaNet-11 to VanillaNet-13
- [ ] ImageNet-1K Pretrained Weights of VanillaNet-11 to VanillaNet-13
- [ ] Downstream Transfer (Detection, Segmentation) Code

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | #params(M) | FLOPs(B) | Lacency(ms) | Acc(%) | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| VanillaNet-5 | 15.5 | 5.2 | 1.61 | 72.49 | [model]() |
| VanillaNet-6 | 32.5 | 6.0 | 2.01 | 76.36 | [model]() |
| VanillaNet-7 | 32.8 | 6.9 | 2.27 | 77.98 | [model]() |
| VanillaNet-8 | 37.1 | 7.7 | 2.56 | 79.13 | [model]() |
| VanillaNet-9 | 41.4 | 8.6 | 2.91 | 79.87 | [model]() |
| VanillaNet-10 | 45.7 | 9.4 | 3.24 | 80.57 | [model]() |
| VanillaNet-11 | 50.0 | 10.3 | 3.59 | 81.08 | - |
| VanillaNet-12 | 54.3 | 11.1 | 3.82 | 81.55 | - |
| VanillaNet-13 | 58.6 | 11.9 | 4.26 | 82.05 | - |
| VanillaNet-13-1.5x | 127.8 | 26.5 | 7.83 | 82.53 | - |
| VanillaNet-13-1.5x&dagger; | 127.8 | 48.5 | 9.72 | 83.11 | - |

## Installation

The results are produced with `torch==1.10.2+cu113 torchvision==0.11.3+cu113 timm==0.6.12`. Other verions might also work.

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


### License
This project is released under the MIT license. Please see the [LICENSE](License) file for more information.
