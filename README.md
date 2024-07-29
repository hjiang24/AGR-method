# Adaptive Gradient Regularization

## [An Adaptive Gradient Regularization Method](https://arxiv.org/abs/2407.16944)

***

## Introduction
* Adaptive gradient regularization (AGR) is a new optimization technique based on the gradient magnitude in a gradient vector, which normalizes the gradient vector in all dimensions as a coefficient vector and subtracts the product of the gradient and its coefficient vector by the vanilla gradient. It can be viewed as an adaptive gradient clipping method. We show that the AGR can improve the loss function Lipschitzness with a more stable training process and better generalization performance. AGR can be embedded into vanilla optimizers such as Adan and AdamW with only three lines of code. To obtain the codes, please refer to the [AGR](https://github.com/hjiang24/AGR-method/blob/master/AGR.py).

<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/gradient.png" height="45%" width="45%" alt="Illustration of the GC operation on gradient matrix/tensor of weights in the fully-connected layer (left) and convolutional layer (right)."/></div>
