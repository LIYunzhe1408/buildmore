

## Exercise 1: Backprop step by step
To better understand how gradients flow through the neural network, we carefully scrutinize the backpropgation process.
```python
emb = C[Xb]
embact = emb.view(emb.shape[0], -1)
# Linear layer 1
hprebn = embact @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n * hprebn.sum(dim=0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1) * bndiff2.sum(dim=0, keepdim=True) # Bessel's correction: dividing by n-1
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact)
# Linear layer 2
logits = h @ W2 + b2 # Output layer
# Cross entropy loss
logit_maxes = logits.max(dim=1, keepdim=True).values
norm_logits = logits - logit_maxes # Subtract maxes for numerical stability
# End up exponetiating logits, if not norm logits, will cause numerical issue when too positive.
counts = norm_logits.exp()
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv # !!! Shape is different. So broadcasting -> times. When a node is used several times, the gradients should be summed up.
logprobs = probs.log()
loss = -logprobs[range(batch_size), Yb].mean()
```
1. For the derivative of `xxx.mean()`
   1. `xxx.mean()` <= $\frac{(prob_{1,13}+prob_{2,2}+prob_{3,15}+prob_{4，1}+...)}{n}$ probs are from the selected position at each row. Others should be 0
   2. Prepare a `torch.zeros_like(logprob)`
   3. Assign `1.0/n` to selected position
2. Remember the chain rule.
3. Before calculate the gradients, check the shape first.
4. If the variable has been broadcast, it means a same variable has been used several time. So perform `xxx.sum(dim=x)` keepdim or not is case by case.
5. Make sure if the variable depends multiple branch, if so summing them up.
6. For `xxx.sum()`
   1. `xxx.sum()` <= $a_{11}+a_{12}+a_{13}=b_{11}$
   2. The derivative for each $a$ is 1.
   3. Prepare `torch.ones_like()` then chain rule.
7. For `xxx.max(dim=y)`,
   1. `dxxx = F.one_hot(xxx.max(dim=y, keepdim=False).indices, xxx.shape[y]) * dlogit_maxes` 
8. For matrix multiplication, derive a simple case by yourself would help understand. Also the shape info help as well.
   1. `logits = h @ W2 + b2`
   2. `dh = dlogits @ W2.T`
   3. `dW2 = h.T @ dlogits`
   4. `db2 = dlogits.sum(dim=0)`


## Cross entropy backward pass
In practice, we will not separate out loss calculation into multiple lines as small atomic pieces and we backproped them individually. Mathematically, some items can be cancelled and the expression can be shortened.
```python
# Cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(dim=1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(batch_size), Yb].mean()
```
