# Activations Gradients and BatchNorm
We are now taking in a few characters in the past as context and feeding them into *Multi-Layer Perceptions* which is a character level model and is proposed by Bengio et al.. Then we can generate the next character in the sequence. Before we build more complex model like RNN which is not optimizable. We would like to first scrutinize how
* activations in the forward pass
* gradients in the backward pass

behave and look like during the training process. It helps to stabilize the training of your neural networks and better understand larger and more complex models in the future.

We did not try to beat our previous performance by introducing the BatchNorm layer.
![alt text](./figures/activation%20distribution.png)
![alt text](./figures/gradients%20distribution.png)
![alt text](./figures/weights%20distribution.png)
![alt text](./figures/updated%20grad_data%20ratio%20over%20time.png)


## Initialization
> Before you start, take some time to understand the wrapped-up starter code which is basically cleaned up based on MLP.

Previously in MLP, the loss of the first iteration is high and rapidly drop which means the model is very confidently wrong and the initialization is messed up.

We expect to have uniform distribution and assign equal prob to each.
### Manually calculate scale
1. loss
    * For last layer, debug `logits`, `probs`, and `loss`. We will see if we want to have `uniformly distributed` not normally distributed, we need to make it near 0 but not exact 0.
      * `b` -> * 0
      * `w` -> * 0.01
2. tanh
    * The activations of the hidden state has many -1 and 1
    *  => Saturated. Because `h=emb@w+b` has broad range and will be set to -1 and 1 when squashing them by `tanh`.
    *  Why we dont want -1 and 1? when `backward`, the gradients of tanh is defined as `(1-t**2)*out.grad`, 1 and -1 will make it 0.
    *  sigmoid, tanh, ReLU suffers this saturation wile Leaky ReLU, ELU not.
       *  `w` -> *0.2
       *  `b` -> *0.01
       *  To shrink the value range of h
```python
See shape of h
stretch it to one line by h.view
tolist.() -> List[float]
plt.hist(list, 50 bins)
plt.imshow(h.abs()>0.99, cmap=gray) # if all columns white => dead neuron, permanent brain damage
```
3. Linear layers
    * When we `y=x@w`, see `x.mean(), x.std()` and `y.mean(), y.std()`, value of `y` is expanding. How to scale `w` to preserve x's distribution.
    * Kaiming Init. `w = torch.randn(fan_in, fan_out) / fan_in ** 0.5 * gain` for linear layers except last layer. $w\times{\sqrt{\frac{\text{gain}}{\text{fan\_in}}}}$ where different non-linearity has different gain value. For `tanh` it's $\frac{5}{3}$
    * In this way, the activations will well behaved not to infinity nor to 0.
### BatchNorm
But for larger and deeper neural networks, it's impossible to calculate this scale for all. BatchNorm is one of the first modern innovation. Again, we expect the `hpreact` is neither too small nor too large s.t. tanh will neither do nothing nor saturated. So we want `hpreact` to be roughly gaussian with 0 mean and 1 std, at least in initialization. 

BatchNorm is here to normalize it to be roughly gaussian at initialization and learn to shift and scale during the training.
```python
mean = hpreact.mean(0, keepdim=True)
std = hpreact.std(0, keepdim=True)
xhat = (hpreact - mean) / std
h = bngain(gamma) * x_hat + bnbias(beta) # Shift and scale. We dont want always to be gaussian during training
```
In paper, it was strictly calculated as 
* $\mu_B\lArr \frac{1}{m}\sum_{i=1}^m{x_i}$
* $\sigma^2_B\lArr \frac{1}{m}\sum_{i=1}^m{(x_i-\mu_B)^2}$
* $\hat{x}_i\lArr \frac{x_i-\mu_B}{\sqrt{\sigma^2_B+\epsilon}}$


Several notes for BatchNorm:
1. Always append a BN layer after linear or convolutional layer to control the scale of activations.
2. However, BN has a terrible cost and introduce a little bit entropy as it couples other samples in that randomly sampled batch. So the function is of all the other samples that happen to come for the same ride rather than a single sample => LayerNorm, GroupNorm to save.
3. Calibration is used to handle one example input for BatchNorm as it's a normalization for one batch. Here we will use the overall mean and std for that set. We get it either by two-stage way (training and mean+std) or calculating running mean and std during training.
4. As h is updated by x_hat which is calculated by mean and std, we eliminating bias in h when calculating means. So `bnbias` is exactly doing the job `bias` was doing. So `bias` for `h` can be eliminated.

## PyTorch-ifying Code
ResNet is stacked by repeated ResNet blocks, internally these blocks are the same structure. Its structure is `x=>conv+BN=>conv+BN=>.....`.

Insights from `torch.nn`
* `nn.linear(fan_in, fan_out, bias)`. First two args are used to initialize weights uniformly with the scale of $(-\frac{1}{\sqrt{\text{fan\_in}}}, \frac{1}{\sqrt{\text{fan\_in}}})$
* `nn.BatchNorm(num_features, eps, momentum, affine, track_running_stat)`
  * num_features: feature dim
  * eps: for avoiding divide by 0 in x_hat calculation
  * momentum: 0.999 & 1-0.999 to update running_mean and running_std
  * affine: if scale and shift or not(gemma and beta)
  * track: estimate mean and std two-stagely
> Conv is used for detect spatial structure but is still a linear multiplication & bias on patches. `nn.conv2D(----, bias=False)` in ResNet code where bias is false is because of BN reason explained in notes for BN.

See how we pytorch-ify the code in notebook. We basically imitate how PyTorch did by hand to modularize blocks for `Linear`, `BatchNorm1D`, and `Tanh` and use them to construct the neural network.
```python
class Linear:
    def __init__(self)

    def __call__(self)

    def parameters(self)
```

## Diagnostic Tool
Visualize intermediate activations, gradients and other statistics throughout the model to scrutinize whether your network is training on the right track. 

### Activation
Based on previous part, we first investigate the activations after `tanh`. `if isinstance(layer, Tanh)`. In this part, we want to see if this layer is saturated by showing `Saturated: (layer.out > 0.97).float().mean()` and plotting the histogram of `layer.out` by `hy, hx = torch.histogram(layer.out)`.
> Try what if not scale the gain by $\frac{5}{3}$ at init. 1 => too small 3 => saturated. Gain is necessary to expanding activations with several tanh squashing func

### Gradient
Then similar viz for gradients. Just show std and mean and plot `layer.out.grad`. When gain is $\frac{5}{3}$, grad at every layer roughly the same:). When it's too small, more value at first layer will be 0. Too large, as activation saturated, more value at last layer will be 0.

Interesting, if we remove all `tanh`, we will have many linear layers stacked up. However, they will collapse to one big linear layer and has limited expressive ability. It's just the linear transformation in forward pass. `Tanh` transforms linear functions to have the ability to approximate any arbitrary functions.

### Parameters
The scale of gradients and values as updates matters the most.

We show the shape, mean, std, and grad:value ratio of `p for p in parameters` because as we update `p += -lr * p.grad` we dont want the grad larger than p. As we can see, the ratio for last layer is not good: it was trained 10 times faster than others as std 10 times than others but it somehow become stable after 1000 training steps.

However, this ratio is not informative but after updated one is better which actually change the tensors. So we instead plot $\frac{\text{lr}\times{\text{p.grad.std()}}}{\text{p.data.std()}}$

Notes:
* The plot over time(i.e. 1000 iterations in notebook) should be around 1e-3. A bit above is okay but below it means learning too slow. It's a good viz for tuning lr.

## Conclusion
1. Introduce Batch Normalization which one of the first modern innovation that helped stabilize training very deep neural networks. How the BatchNorm works? How would it be used in neural networks?
2. PyTorch-ify the code and wrap them into modules like `Linear`, `BatchNorm1D`, and `Tanh`. They are layers or modules that can be stacked up like lego blocks. These layers actually exist in `torch.nn`.
3. Diagnostic tools you would use to understand whether your neural network is in the good state dynamically. Statistics and histograms of the forward pass activation and backward pass gradients. And then we're looking at the weights that are going to be updated as part of the stochastic gradient descent, `mean`, `std`, and `grad:data` ratio are scrutinized. And even better the updated grad to data ration. Typically we don't actually look at the them as a snapshot frozen in time at some iteration. Typically people look at this as a over time like 1000 iterations. You can have a tool or an idea whether your training is on the right track.

We did not try to beat our previous performance by introducing the BatchNorm layer. Our current performance is not bottlenecked by the optimization which is what batchnorm helping with. The performance at this stage is bottlenecked by context length. We may use more powerful architectures like RNN and Transformers in order to further push the locked probabilities.