
# High-Performance Neural Network Benchmark in Parallel with SIMD and Modern C# Optimization on MNIST

This repository is based on the idea of creating a benchmark that better illustrates what a normal-sized neural network on a home PC can achieve. For this purpose, the MNIST dataset consisting of 28 by 28 pixels is used. The neural network has 3 layers with the following architecture: 784-100-100-10
The easiest way to understand this work is by checking out the demo.

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/benchmark.png?raw=true">
</p>

The demo first checks if the dataset is located under the specified path on the PC. If not, it is automatically loaded from my GitHub repository. After normalizing the data and creating the hyperparameters, the neural network is initialized and training begins. Please note that when executing multiple cores in parallel, the outcomes can be unpredictable. This implies that the results may vary with each training run, although it is also possible for the results to remain consistent.

After conducting numerous benchmarks, the following result emerged: Training for 50 epochs with 60,000 examples, utilizing various optimization techniques, took less than 2 seconds on an AMD 5600X processor with 6 cores. Unlike training, where the results can vary, the test results remain consistent during parallel operations. Although the test accuracy of 98% is good, the test frames per second were even more interesting and exceeded 2.1 million. An unbelievably high value, and especially considering that my system is quite mid-range. An core count increased from 6 to 16 with newer CPU generations should easily calculate 3x faster in the same benchmark. Let's see how that became possible.


## [Neural Network Feature List](#neural-network-feature-list)

- [Mini-Batch Training](#mini-batch-training)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [From Scratch Concept](#from-scratch-concept)
- [Modern C# Optimization](#modern-c-optimization)
- [Parallel Execution](#parallel-execution)
- [SIMD Support](#simd-support)
- [Layer-Wise Propagation](#layer-wise-propagation)
- [ReLU Pre-activation](#relu-pre-activation)
- [Backprop Probability Training](#backprop-probability-training)
- [One Loop Backprop](#one-loop-backprop)
- [Layer-wise Gradients](#layer-wise-gradients)
- [Cache Locality](#cache-locality)

## Mini-Batch Training


<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/mini-batch.png?raw=true">
</p>

A pillar in the training process is the batch size. First an example is taken, then the forward propagation is executed, which calculates the activations of the output neurons for each layer in the neural network. Then the error is determined by output - target = error. Which forms our output gradients that can be multiplied by their weights to calculate the further gradients until all layers have been backpropagated. 

We could update the weights now, but instead of directly modifying the weights, we follow a different strategy. We store each correction value in a delta array with the same length of the weights, enabling us to accumulate a batch of these correction values. This approach has been found to yield better results compared to the shortcut. However, before we can update the weights, we need to determine the batch size, which represents the number of examples processed together. Once we have a batch filled with these correction values, we can proceed to update the weights accordingly.

Let's understand the calculation cost of the training process of each example, which consists of (FF: activations = 1) + (BP: gradients = 1 + deltas = 1) + (Update: SGD = 1). The total cost is approximately 4. It's important to grasp the cost with different batch sizes. For example, a batch size of 1 would have a cost of 4, as we update the weights after computing the gradients, which only would cost 3. With a batch size of 2, the cost would be 3.5. In the demo, a batch size of 800 is used, which is considered high. However, this high batch size reduces the overall cost while improving training efficiency. To learn more about the impact of the batch size and its results, you can check out this [repository](https://github.com/grensen/multi-core#batchsize-800-with-net-7) with different batch sizes.

## Stochastic Gradient Descent


<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/SGD.png?raw=true">
</p>

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

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/nick_stephen.png?raw=true">
</p>

[Why is .NET so Insanely Fast? with Stephen Toub | Keep Coding Podcast #7](https://www.youtube.com/watch?v=Hxfu_KEa4uA) 

When it comes to modern C#, this article [Performance Improvements in .NET 7](https://devblogs.microsoft.com/dotnet/performance_improvements_in_net_7/) by Stephen Toub is the first place to look. Perhaps the most relevant new feature are Span arrays, which are used everywhere in my code. Furthermore, there have been significant improvements in internal functions like Span1.CopyTo(Span2). 

## Parallel Execution

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/multi-core.png?raw=true">
</p>

Parallel execution in that case means, that each available core in a CPU is assigned to calculate an example. However, this causes problems that only become apparent during parallel execution. The floating-point precision is not very accurate, which means that inaccuracies occur in the calculations, as can be seen in this [floating point issue example](https://github.com/grensen/multi-core#floating-point-issues) where discrepancies occur in about 10% of the calculations. Each time, the different executions lead to different delta values. If all are equally affected, the results remain the same. However, parallel execution tends to follow randomness, leading to the frustrating lack of reproducibility in our results. Techniques like weight decay can somewhat mitigate this effect, but the effect still exists, and perhaps more luck is involved in successful training than we would like. It remains an open challenge. However, 6 cores compute almost 6 times more than 1 core in my case. 
  
## SIMD Support

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/simd_locuza.png?raw=true">
</p>

[(Image credit: @Locuza_ via Twitter)](https://twitter.com/Locuza_/status/1454152714930331652/photo/2) 
  
SIMD stands for Single Instruction Multiple Data and is a technique where a single instruction is applied to multiple data simultaneously. Finding the figure was a bit challenging, but it beautifully illustrates the presence of SIMD units on each CPU core. These SIMD units can process a vector of 8 values in my case, or possibly a vector of only 4 values, depending on the specific CPU architecture. Probably the CPU you're running the demo on will also support SIMD, so it's standard for the benchmark. As far as I know, the vector class I use can also support future CPU generations that can handle larger vectors like 16 or 32. That would be pretty cool. However, there are different approaches to achieve what we want. The method I use is similar to a great example on [Stack Overflow from aepot](https://stackoverflow.com/questions/71627141/fastest-way-to-multiply-and-sum-add-two-arrays-dot-product-unaligned-surpris/72760499#72760499) that demonstrates how to utilize SIMD vectors as references without copying them. Code sorcery!?!? It is heavily used in the code!
  
## Layer-wise Propagation
  
<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/ff_layer-wise.gif?raw=true">
</p>  
  
This animation is not entirely accurate, as it explicitly shows the activation, which is obtained on the fly in the implementation. Additionally, the ReLU function remains exclusive for this task as it is the only function capable of deactivating neurons so far. It may not be obvious, but by iterating through the network in this way, we open up a number of advantages that come into play.  
## ReLU Pre-activation

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/relu.png?raw=true">
</p>

Pre-activation means that we take an input neuron and then use the general ReLU activation, we do this for every input neuron on every layer. This has the advantage of saving a lot of computation. The disadvantage is that this implementation cannot handle negative input neurons on the first layer, we would have to modify the code slightly for that.

Let's go through how to evaluate the effect, starting with weights that are approximately 50% positive and 50% negative. Since the ReLU activation function allows only positive values, and the weights are connected to the input signals, the dot product, which can also be called the weighted sum, is likely to produce 50% activated neurons and 50% deactivated neurons. For example, from 784 * 100 = 78400 weights on the first layer, theoretically only half with 39200 weights will be computed. This of course continues on each layer. With training and competitive learning rates, the activation level decreases further depending on the strength of the learning rate. It is not uncommon for only 20% of ReLU activations to be active in trained networks. 

It gets a bit mystical when I keep the learning rate very low, then the activation level can even go up to 100% if all neurons have enough activation possibilities to be trained. Crazy stuff!

## Backprop Probability Training

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/backprop_probability.png" />
</p>

Looking back at the costs, the backprop process is the most expensive part of the computation with a cost of two. The idea is to take the softmax activation of the last output layer, which produces probabilities of 0-100% from our weighted sums. Then a probability check is performed to see if the activated output neuron of the target class is below the defined threshold, and only then is backprop executed. 

Assuming outputNeuron = 1 and target = 1, the calculation would look like this: outputNeuron - target = error = 0. An error of 0 means that we have nothing to learn here, since the difference between what we want and what comes out is zero. But even if the error is only 1%, it has not proven to be useful to tweak these examples any further, rather experience has shown that the generalization will continue to improve. 

Backprop doesn't care about the error, it would just go through and cause a cost of 2 for every example we can already perfectly predict. In practice, however, after only one epoch, about 50% of the examples are predicted above the threshold. With one more line of code, the training time is practically halved from 4 to 2 seconds, so the probability threshold for backrop is an important feature. 

~~~cs
static bool Train(Span<float> sample, byte target, Net NN, float[] deltas)
{
    Span<float> neurons = stackalloc float[NN.neuronLen];

    int prediction = Eval(sample, neurons, NN.net, NN.weights);

    Softmax(neurons.Slice(neurons.Length - NN.net[^1], NN.net[^1]));

    if (neurons[NN.neuronLen - NN.net[^1] + target] < 0.99) // the magic line
        Backprop(NN.net, NN.weights, neurons, deltas, target);

    return prediction == target;
}
~~~

## One Loop Backprop

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/one-loop.png?raw=true">
</p>

One loop backprop means that we can take the output gradients and multiply them by the input neuron to get the deltas, and also multiply them by the weights to sum the input gradients. 
This is only possible because we are computing the layer layer-wise, which means input to outputs, not perceptron-wise, where we compute inputs to output. 

But there is also a drawback, because the input gradients are also computed, which are not needed in this implementation. So the code could be further optimized by unrolling the loop for the first layer input to hidden1. But then the code would be more and the input gradients on the first layer would be not ready for a CNN, so they will be computed as well. 

## Layer-Wise Gradients

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/layer-wise-gradients.png?raw=true">
</p>

Understanding why one loop backprop works is important and not easy. In contrast to two loop backprop, which first computes all gradients and then all deltas, with one loop backprop we can save memory and make the network even faster by storing only the input and output gradients of each layer. 

Only the code most important code parts:
~~~cs
    // caution - input gradients are computed! (for CNN e.g.)
    Span<float> outputGradients = stackalloc float[net[^1]]; // right

    // output error gradients, hard target as 1 for its class
    for (int r = neurons.Length - net[^1], p = 0; r < neurons.Length; r++, p++)
        outputGradients[p] = target == p ? 1 - neurons[r] : -neurons[r];

    // compute gradients and deltas for each layer in one loop, hot!!!
    for (int j = neurons.Length - net[^1], k = neurons.Length, m = weights.Length, i = net.Length - 2; i >= 0; i--)
    {
        int right = net[i + 1], left = net[i];
        k -= right; j -= left; m -= right * left;

        Span<float> inputGradients = stackalloc float[left];
        for (int l = 0, w = m; l < inputGradients.Length; l++, w += right)
        {
            // backprop calculations
        }

        if (i != 0) // dirty! but seems faster without i == 0
        {
            if (outputGradients.Length < inputGradients.Length) 
                outputGradients = stackalloc float[inputGradients.Length]; // faster but dirty? 
            inputGradients.CopyTo(outputGradients); // overwrite used array
        }
    }  
}
~~~

Note that after this optimization we no longer copy the input gradients for the first layer, which seems to be a bit faster. Also note, since feedforward and backprop can use the same forward pass for simplicity, the code remains as it is. But the input-output gradients idea can be used with the same trick for inference in production, since not all activated neurons need to be stored there.

## Cache Locality

<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/blob/main/figures/cache_locality.png?raw=true">
</p>





