
# High-Performance Neural Network Benchmark in Parallel with SIMD and Modern C# Optimization on MNIST

This repository is based on the idea of creating a benchmark that better illustrates what a normal-sized neural network on a home PC can achieve. For this purpose, the MNIST dataset consisting of 28 by 28 pixels is used. The neural network has 3 layers with the following architecture: 784-100-100-10
The easiest way to understand this work is by checking out the demo.

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/benchmark.png?raw=true">
</p>

The demo first checks if the dataset is located under the specified path on the PC. If not, it is automatically loaded from my GitHub repository. After normalizing the data and creating the hyperparameters, the neural network is initialized and training begins. Please note that when executing multiple cores in parallel, the outcomes can be unpredictable. This implies that the results may vary with each training run, although it is also possible for the results to remain consistent.

After conducting numerous benchmarks, the following result emerged: Training for 50 epochs with 60,000 examples, utilizing various optimization techniques, took less than 2 seconds on an AMD 5600X processor with 6 cores. Unlike training, where the results can vary, the test results remain consistent during parallel operations. Although the test accuracy of 98% is good, the test frames per second were even more interesting and exceeded 2.1 million. An unbelievably high value, and especially considering that my system is quite mid-range. An core count increased from 6 to 16 with newer CPU generations should easily calculate 3x faster in the same benchmark. Let's see how that became possible.


## [Neural Network Feature List](#neural-network-feature-list)

- [Mini-batch Training](#mini-batch-training)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [From Scratch Concept](#from-scratch-concept)
- [Modern C# Optimization](#modern-c-optimization)
- [Parallel Execution](#parallel-execution)
- [SIMD Support](#simd-support)
- [Layer-wise Propagation](#layer-wise-propagation)
- [ReLU Pre-activation](#relu-pre-activation)
- [Backprop Probability Training](#backprop-probability-training)
- [Layer-wise Gradients](#layer-wise-gradients)
- [One Loop Backprop](#one-loop-backprop)
- [Cache Locality](#cache-locality)

## Mini-batch Training

A pillar in the training process is the batch size. First an example is taken, then the forward propagation is executed, which calculates the activations of the output neurons for each layer in the neural network. Then the error is determined by output - target = error. Which forms our output gradients that can be multiplied by their weights to calculate the further gradients until all layers have been backpropagated. 

We could update the weights now, but instead of directly modifying the weights, we follow a different strategy. We store each correction value in a delta array with the same length of the weights, enabling us to accumulate a batch of these correction values. This approach has been found to yield better results compared to the shortcut. However, before we can update the weights, we need to determine the batch size, which represents the number of examples processed together. Once we have a batch filled with these correction values, we can proceed to update the weights accordingly.

Let's understand the calculation cost of the training process of each example, which consists of (FF: activations = 1) + (BP: gradients = 1 + deltas = 1) + (Update: SGD = 1). The total cost is approximately 4. It's important to grasp the cost with different batch sizes. For example, a batch size of 1 would have a cost of 4, as we update the weights after computing the gradients, which only would cost 3. With a batch size of 2, the cost would be 3.5. In the demo, a batch size of 800 is used, which is considered high. However, this high batch size reduces the overall cost while improving training efficiency. To learn more about the impact of the batch size and its results, you can check out this [repository](https://github.com/grensen/multi-core#batchsize-800-with-net-7) with different batch sizes.

## Stochastic Gradient Descent

To update the weights, we use an optimizer, in this case, stochastic gradient descent (SGD) with momentum. Each weight goes through the same process, after the delta value for each weight has been accumulated in a batch, the calculation is newWeight = weight + learningRate * delta. Delta multiplied by learningRate is only a very small part in the right direction, so we hope to get better results with each update step. The new delta is calculated as newDelta = delta * momentum, and contains some of the correction of the last delta. Although momentum and more sophisticated ideas are often used, there are also efforts to do without momentum, which is sometimes better. In addition, learning rate and momentum are decreased each epoch with a factor smaller than 1. 

Even if SGD is not perfect, it often offers a very good compromise and is used here, another important point why the neural network is so fast.

## From Scratch Concept

<p align="center">
  <img src="https://github.com/grensen/how_to_build/raw/main/figures/network_intuition.png?raw=true">
</p>

The figure above shows how the neural network works. The most fundamental change is, instead of the perceptron-wise execution also known as dot-product, which calculates all inputs to one output. Here we do the opposite by calculating one input to all outputs. This allows us to exploit the full potential of the ReLU activation function. Think about, every computation in the blue rectangle can be discard in that way if the activation level is below or equal to zero.

To build a deep learning system, it is essential to understand and follow these basic steps:

- Allocate memory for the system.
- Initialize weights.
- Perform forward propagation.
- Perform backpropagation.
- Update the weights.

There are a few more details, but if we start thinking about building a new network, those would be the critical points. Surprisingly, iterating through the network only once is sufficient. You can easily replicate this code to carry out the necessary mathematical operations for each of the five steps. This not only prevents a lot of mistakes that would otherwise be made, but also forms a solid basis for taking further steps, and we will take a lot of them.

But C# instead of Python you might think, which is by far the most popular language for building neural networks. Python is predominantly associated with libraries that heavily rely on other programming languages like C++ or Rust. However, our computations require deep changes, necessitating the flexibility to observe and modify every aspect. While Python can also be employed for this purpose, it tends to exhibit sluggish performance, especially when involving loops.

C# may seem unusual to the experienced reader, but it is important to note that in 2023, C# has made significant advancements in its latest version, .NET 7. These advancements, driven by the efforts of many talented individuals, have propelled C# to new heights. Moreover, C# is an open-source language, fostering collaboration and innovation within the development community. Where C# stands today can be seen very well in this test: [How Much Memory for 1,000,000 Threads in 7 Languages | Go, Rust, C#, Elixir, Java, Node, Python](https://www.youtube.com/watch?v=WjKQQAFwrR4). Or just through this benchmark. 

However, if anyone believes that C# is outdated, I invite them to build the demo in a more advanced language like Rust or Zig and attempt to outperform my demo on a similar PC.

## Modern C# Optimization

When it comes to modern C#, this article [Performance Improvements in .NET 7](https://devblogs.microsoft.com/dotnet/performance_improvements_in_net_7/) by Stephen Toub is the first place to look. Perhaps the most relevant new feature are <span> arrays, which are used everywhere in my code. Furthermore, there have been significant improvements in internal functions like Span1.CopyTo(Span2). 

## Parallel Execution

Parallel execution in that case means, that each available core in a CPU is assigned to calculate an example. However, this causes problems that only become apparent during parallel execution. The floating-point precision is not very accurate, which means that inaccuracies occur in the calculations, as can be seen in this [floating point issue example](https://github.com/grensen/multi-core#floating-point-issues) where discrepancies occur in about 10% of the calculations. Each time, the different executions lead to different delta values. If all are equally affected, the results remain the same. However, parallel execution tends to follow randomness, leading to the frustrating lack of reproducibility in our results. Techniques like weight decay can somewhat mitigate this effect, but the effect still exists, and perhaps more luck is involved in successful training than we would like. It remains an open challenge. However, 6 cores compute almost 6 times more than 1 core in my case. 
  
## SIMD Support

  
  
## Layer-wise Propagation

## ReLU Pre-activation

## Backprop Probability Training

## Layer-wise Gradients

## One Loop Backprop

## Cache Locality





