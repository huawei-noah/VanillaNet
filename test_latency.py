#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch.nn as nn
import torch
import torchvision
import time

import models.vanillanet


if __name__ == "__main__":
    from timm.data import create_dataset, create_loader
    dataset_val = create_dataset(name='', root='/data/imagenet/', split='validation', is_training=False, batch_size=1)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    size = 224
    data_loader_val = create_loader(dataset_val, input_size=size, batch_size=1, is_training=False, use_prefetcher=False)
    
    net = vanillanet_5().cuda()
    net.eval()
    print(net)
    for img, target in data_loader_val:
        img = img.cuda()
        for i in range(5):
            net(img)
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            for i in range(1000):
                net(img)
                torch.cuda.synchronize()
        print((time.time() - t))

        n_parameters = sum(p.numel() for p in net.parameters())
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        from torchprofile import profile_macs
        macs = profile_macs(net, img)
        print('model flops (G):', macs / 1.e9, 'input_size:', img.shape)
        
        break
