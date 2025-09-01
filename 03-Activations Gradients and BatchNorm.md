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
    * Kaiming Init. `w = torch.randn(fan_in, fan_out) / fan_in ** 0.5`
    * In this way, the activations will well behaved not to infinity nor to 0.
### BatchNorm
But for larger and deeper neural networks, it's impossible to calculate this scale for all.
## PyTorch-ifying Code

## Diagnostic Tool

## Conclusion
1. Introduce Batch Normalization which one of the first modern innovation that helped stabilize training very deep neural networks. How the BatchNorm works? How would it be used in neural networks?
2. PyTorch-ify the code and wrap them into modules like `Linear`, `BatchNorm1D`, and `Tanh`. They are layers or modules that can be stacked up like lego blocks. These layers actually exist in `torch.nn`.
3. Diagnostic tools you would use to understand whether your neural network is in the good state dynamically. Statistics and histograms of the forward pass activation and backward pass gradients. And then we're looking at the weights that are going to be updated as part of the stochastic gradient descent, `mean`, `std`, and `grad:data` ratio are scrutinized. And even better the updated grad to data ration. Typically we don't actually look at the them as a snapshot frozen in time at some iteration. Typically people look at this as a over time like 1000 iterations. You can have a tool or an idea whether your training is on the right track.

We did not try to beat our previous performance by introducing the BatchNorm layer. Our current performance is not bottlenecked by the optimization which is what batchnorm helping with. The performance at this stage is bottlenecked by context length. We may use more powerful architectures like RNN and Transformers in order to further push the locked probabilities.