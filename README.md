# Adaptive Gradient Regularization

## [An Adaptive Gradient Regularization Method](https://arxiv.org/abs/2407.16944)

***

## Introduction
* Adaptive gradient regularization (AGR) is a new optimization technique based on the gradient magnitude in a gradient vector, which normalizes the gradient vector in all dimensions as a coefficient vector and subtracts the product of the gradient and its coefficient vector by the vanilla gradient. It can be viewed as an adaptive gradient clipping method. We show that the AGR can improve the loss function Lipschitzness with a more stable training process and better generalization performance. AGR can be embedded into vanilla optimizers such as Adan and AdamW with only three lines of code. To obtain the codes, please refer to the [AGR](https://github.com/hjiang24/AGR-method/blob/master/AGR.py).

<div align="center">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/AGR_1.png" alt="图片1" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/AGR_2.png" alt="图片2" width="300">
</div>

* We conduct the experiments among a series of tasks across multiple networks in Adan and AdamW, which are provided in [Adan_AGR](https://github.com/hjiang24/AGR-method/blob/master/Adan_AGR.py) and [AdamW_AGR](https://github.com/hjiang24/AGR-method/blob/master/AdamW_AGR.py), respectively. It's also easy and similar to embed the AGR method in other popular optimizers such as SGDM, Adam, etc.

## Experiments
***
### Image generation
We conduct image generation based on DDPM accelerated by DDIM among different optimizers. The best performance is achieved by the Adan optimizer with the AGR method on the CIFAR10 dataset. The code is provided in [Image_generation](https://github.com/hjiang24/AGR-method/edit/master/Image_classification). The FID and IS score are shown as below:

|IS(std)        |  10           |   400        | 800         | 1200        | 1600        | 2000        |
| :-----------: | :-----------: | :----:       |:------:     |:-------:    |:-------:    |:-------:    |
| ACProp        | 2.95(0.03)    |7.48(0.07)    |8.30(0.07)   |8.44(0.09)   |9.18(0.16)   |9.83(0.17)   |
| RMSprop       | 3.92(0.04)    |6.94(0.07)    |8.62(0.14)   |8.61(0.05)   |9.25(0.12)   |9.30(0.06)   |
| Adam          | 3.76(0.04)    |7.73(0.13)    |8.44(0.11)   |8.96(0.16)   |9.01(0.09)   |9.16(0.13)   |
| AdamW         | 3.99(0.04)    |7.84(0.08)    |8.91(0.12)   |9.02(0.04)   |9.11(0.09)   |9.18(0.15)   |
| Adan          | 4.31(0.06)    |8.14(0.13)    |8.85(0.10)   |9.18(0.10)   |9.19(0.07)   |9.22(0.11)   |
| Adan(AGR)     | 4.38(0.05)    |8.32(0.10)    |8.86(0.12)   |9.18(0.08)   |9.26(0.13)   |9.34(0.12)   |


|FID            |  10           |   400        | 800         | 1200        | 1600        | 2000        |
| :-----------: | :-----------: | :----:       |:------:     |:-------:    |:-------:    |:-------:    |
| ACProp        | 304.03        |40.06         |28.42        |12.32        |10.12        |9.57         |
| RMSprop       | 313.19        |54.67         |23.56        |10.35        |8.40         |8.10         |
| Adam          | 229.30        |55.12         |21.91        |10.45        |10.27        |9.27         |
| AdamW         | 210.46        |23.90         |17.43        |10.05        |9.11         |9.01         |
| Adan          | 169.32        |14.60         |13.68        |8.64         |8.07         |7.98         |
| Adan(AGR)     | 161.65        |13.48         |12.75        |8.08         |7.85         |7.44         |

### Image classification
We conduct image classification with multiple popular architectures consisting of CNN architectures such as VGG11, ResNet and ConvNext, and transformer architectures such as ViTs, and Swin on AdamW optimizer with and without the AGR method. The following figures are training loss and test accuracy.
<div align="center">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/resnet18_loss.png" alt="图片1" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/resnet18_accuracy.png" alt="图片2" width="300">
</div>

<div align="center">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/swin_loss.png" alt="图片1" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/vgg_loss.png" alt="图片2" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/Tiny_ViT_loss.png" alt="图片1" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/swin_accuracy.png" alt="图片2" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/vgg_accuracy.png" alt="图片1" width="300">
    <img src="https://github.com/hjiang24/AGR-method/blob/master/figures/Tiny_accuracy.png" alt="图片2" width="300">
</div>

### Language representation
We conduct sentence order prediction (SOP) and masked language model (MLM) by Albert on optimizer AdamW with and without the AGR method. The following table shows the SOP accuracy with different weight decay.
|Weight decay   |  0            |   2e-4       | 5e-4        | 1e-3        | 1e-2        |
| :-----------: | :-----------: | :----:       |:------:     |:-------:    |:-------:    |
| AdamW         | 80.02         |82.14         |82.01        |82.94        |82.73        |
| AdamW(AGR)    | 80.55         |84.19         |84.47        |83.21        |83.06        |
