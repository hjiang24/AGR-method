# Adaptive Gradient Regularization

## [An Adaptive Gradient Regularization Method](https://arxiv.org/abs/2407.16944)

***

## Introduction
* Adaptive gradient regularization (AGR) is a new optimization technique based on the gradient magnitude in a gradient vector named , which normalizes the gradient vector in all dimensions as a coefficient vector and subtracts the product of the gradient and its coefficient vector by the vanilla gradient. It can be viewed as an adaptive gradient clipping method. We show that the AGR can improve the loss function Lipschitzness with a more stable training process and better generalization performance. AGR is very simple to be embedded into vanilla optimizers such as Adan and AdamW with only three lines of code. Please refer to the [AGR](https://github.com/hjiang24/AGR-method/blob/master/AGR.py) to obtain the codes.
