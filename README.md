# Make More from Biagram to Transformer

## [Biagram](./01-Biagram.md)
We introduce bigram character level language model. We saw how we can train the model, how we can sample from the model, and how we can evaluate the quality of the model using the negative log likelihood loss.

## [MLP](./02-MLP.md)
Based on the single layer network built before, we step one more further to build an MLP which basically add hidden layers between input and output. We show that the performance is much better.

## [Activations, Gradients, and BatchNorm](./03-Activations%20Gradients%20and%20BatchNorm.md)
Introduce BatchNorm to stablize the training process, and viz tools to plot activation, gradients, and some ratios dynamically. Also we PyTorch-ifying the code to modules for future extension.

## [Backprop Ninja](./04-Backprop%20Ninja.md)
We did not use `loss.backward()` and pytorch auto grad, and we estimate gradients ourselves by hand. It gave us a pretty nice diversity of layers to backprop through. It also gave us a pretty nice and comprehensive sense of how these backward passes are implemented and how they work. You will have some intuition about how gradients flow backwards throughthe neural net starting at the loss and how they flow through all the variables and all the intermediate results.