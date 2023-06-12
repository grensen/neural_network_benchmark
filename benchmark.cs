// C# 11 required
using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

PrintUserInfo();

AutoData data = new(@"C:\mnist\"); // get data

// define neural network 
int[] network    = { 784, 100, 100, 10 };
var LEARNINGRATE = 0.0005f;
var MOMENTUM     = 0.67f;
var EPOCHS       = 50;
var BATCHSIZE    = 800;
var FACTOR       = 0.999f;
var SEED         = 12345;
var PARALLEL     = true;

PrintHyperparameters();

Net NN = new(network, SEED);

RunTraining(PARALLEL, data, NN, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

RunTest(PARALLEL, data, NN, 10000);

Console.WriteLine("\nBenchmark completed successfully");
Console.ReadLine();

// TRAINING
static void RunTraining(bool multiCore, AutoData d, Net NN, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    Console.WriteLine($"\nTraining Progress{(multiCore ? " - Parallel" : "")}:\n--------------------------------");

    float[] deltas = new float[NN.weights.Length];

    Stopwatch stopwatch = Stopwatch.StartNew();

    for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel count

        if (multiCore)
            for (int b = 0; b < B; b++)
            {
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    c[x] = Train(d.samplesTrainingF.AsSpan().Slice(x * 784, 784), d.labelsTraining[x], NN, deltas);
                });

                Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, NN.weights.Length), range =>
                {
                    SGD(range.Item1, range.Item2, NN.weights, deltas, lr, mom);
                });
            }
        else // single core
            for (int b = 0; b < B; b++)
            {
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    c[x] = Train(d.samplesTrainingF.AsSpan().Slice(x * 784, 784), d.labelsTraining[x], NN, deltas);

                SGD(0, NN.weights.Length, NN.weights, deltas, lr, mom);
            }

        if ((epoch + 1) % 10 == 0) // info
            PrintInfo("Epoch = " + (1 + epoch).ToString().PadLeft(3) + " |", c.Count(n => n), B * BATCHSIZE, stopwatch);
    }
}
static bool Train(Span<float> sample, byte target, Net NN, float[] deltas)
{
    Span<float> neurons = stackalloc float[NN.neuronLen];

    int prediction = Eval(sample, neurons, NN.net, NN.weights);

    Softmax(neurons.Slice(neurons.Length - NN.net[^1], NN.net[^1]));

    if (neurons[NN.neuronLen - NN.net[^1] + target] < 0.99)
        Backprop(NN.net, NN.weights, neurons, deltas, target);

    return prediction == target;
}
// TESTING
static void RunTest(bool multiCore, AutoData d, Net NN, int len)
{
    Console.WriteLine($"\nTest Results{(multiCore ? " - Parallel" : "")}:\n--------------------------------");

    bool[] c = new bool[len]; // for proper parallel count

    Stopwatch stopwatch = Stopwatch.StartNew();

    if (multiCore)
        Parallel.For(0, len, x =>
        {
            c[x] = Test(d.samplesTestF.AsSpan().Slice(x * 784, 784), d.labelsTest[x], NN);
        });
    else // single core
        for (int x = 0; x < len; x++)
            c[x] = Test(d.samplesTestF.AsSpan().Slice(x * 784, 784), d.labelsTest[x], NN);

    stopwatch.Stop();

    PrintInfo("Test", c.Count(n => n), len, stopwatch, true);
}
static bool Test(ReadOnlySpan<float> sample, byte target, Net NN)
{
    int prediction = Eval(sample, stackalloc float[NN.neuronLen], NN.net, NN.weights);

    return prediction == target;
}
// FF
static int Eval(ReadOnlySpan<float> sample, Span<float> neurons, ReadOnlySpan<int> net, ReadOnlySpan<float> weights)
{
    sample.CopyTo(neurons.Slice(0, 784));

    FeedForward(neurons, net, weights);

    return Argmax(neurons.Slice(neurons.Length - net[^1], net[^1]));
}
static void FeedForward(Span<float> neurons, ReadOnlySpan<int> net, ReadOnlySpan<float> weights)
{
    // Caution, this implementation does not support negative input values!!!

    // for (int i = 0, k = net[0], w = 0; i < net.Length - 1; i++) was more than 10% slower
    for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
    {
        Span<float> outLocal = stackalloc float[net[i + 1]]; // fast temporary output neurons
        var inpLocal = neurons.Slice(k - net[i], net[i]);

        for (int l = 0; l < inpLocal.Length; l++, w = outLocal.Length + w)
        {
            var inpNeuron = inpLocal[l]; // fast temporary input neuron
            if (inpNeuron <= 0) continue; // ReLU input pre-activation

            ReadOnlySpan<float> wts = weights.Slice(w, outLocal.Length);
            ReadOnlySpan<Vector<float>> wtsVec = MemoryMarshal.Cast<float, Vector<float>>(wts); // span to vector reference
            Span<Vector<float>> outVec = MemoryMarshal.Cast<float, Vector<float>>(outLocal); // span to vector reference

            for (int v = 0; v < outVec.Length; v++) // SIMD
                outVec[v] = wtsVec[v] * inpNeuron + outVec[v];

            for (int r = wtsVec.Length * Vector<float>.Count; r < outLocal.Length; r++) // default no bounds
                outLocal[r] = wts[r] * inpNeuron + outLocal[r];
        }
        outLocal.CopyTo(neurons.Slice(k, outLocal.Length)); // stack output neurons
        k = outLocal.Length + k; // stack output id 
    }
}
static int Argmax(Span<float> neurons)
{
    int id = 0;
    float max = neurons[0];
    for (int i = 1; i < neurons.Length; i++)
    {
        float n = neurons[i];
        if (n > max)
        {
            id = i;
            max = n; 
        }
    }
    return id; // prediction
}
// BP
static void Softmax(Span<float> neurons)
{
    float scale = 0;
    for (int n = 0; n < neurons.Length; n++)
        scale += neurons[n] = MathF.Exp(neurons[n]); // activation then sum up

    scale = 1 / scale; // turns division to multiplication

    for (int n = 0; n < neurons.Length; n++)
        neurons[n] *= scale; // probabilities
}
static void Backprop(Span<int> net, Span<float> weights, Span<float> neurons, Span<float> deltas, int target)
{  
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
            var n = neurons[l + j];
            if (n <= 0) continue;

            var wts = weights.Slice(w, right);
            var dts = deltas.Slice(w, right);

            var wtsVec = MemoryMarshal.Cast<float, Vector<float>>(wts);
            var dtsVec = MemoryMarshal.Cast<float, Vector<float>>(dts); 
            var graVec = MemoryMarshal.Cast<float, Vector<float>>(outputGradients);

            var sumVec = Vector<float>.Zero;

            for (int v = 0; v < wtsVec.Length; v++) // SIMD, partial gradient and delta
            {
                var outGraVec = graVec[v];
                sumVec = wtsVec[v] * outGraVec + sumVec;
                dtsVec[v] = n * outGraVec + dtsVec[v];
            }

            // changed float result with Vector.Sum() vs. regular approach
            float sum = Vector.Sum(sumVec);

            for (int r = wtsVec.Length * Vector<float>.Count; r < wts.Length; r++)
            {
                var outGraSpan = outputGradients[r];
                sum = wts[r] * outGraSpan + sum;
                dts[r] = n * outGraSpan + dts[r];
            }
            inputGradients[l] = sum;
        }

        if (i != 0) // dirty! but seems faster without i == 0
        {
            if (outputGradients.Length < inputGradients.Length) 
                outputGradients = stackalloc float[inputGradients.Length]; // faster but dirty? 
            inputGradients.CopyTo(outputGradients); // overwrite used array
        }
    }  
}
static void SGD(int start, int end, Span<float> weights, Span<float> delta, float lr, float mom)
{
    var weightVecArray = MemoryMarshal.Cast<float, Vector<float>>(weights.Slice(start, end - start));
    var deltaVecArray = MemoryMarshal.Cast<float, Vector<float>>(delta.Slice(start, end - start));

    for (int v = 0; v < weightVecArray.Length; v++)
    {
        weightVecArray[v] = deltaVecArray[v] * lr + weightVecArray[v];
        deltaVecArray[v] *= mom;
    }

    for (int w = weightVecArray.Length * Vector<float>.Count + start; w < end; w++)
    {
        weights[w] = delta[w] * lr + weights[w];
        delta[w] *= mom;    
    }
}
// INFO
void PrintUserInfo()
{
#if DEBUG
           Console.WriteLine("Debug mode is on, switch to Release mode");
#endif

    Console.WriteLine($"\nNeural Network Benchmark - {DateTime.Now.Year} | C# (.NET {Environment.Version})\n");
    Console.WriteLine($"CPU: AMD Ryzen 5 5600X | Cores: 6 | Default Speed: 3.7 GHz\n"); // change to your sys
}
void PrintHyperparameters()
{
    Console.WriteLine("Neural Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", network));
    Console.WriteLine("WEIGHTS      = " + network.Zip(network.Skip(1), (prev, curr) => prev * curr).Sum());
    Console.WriteLine("SEED         = " + SEED);
    Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    Console.WriteLine("EPOCHS       = " + EPOCHS);
    Console.WriteLine("FACTOR       = " + FACTOR);
}
static void PrintInfo(string str, int correct, int all, Stopwatch sw, bool showFPS = false)
{
    Console.WriteLine($"{str} Accuracy = {(correct * 100.0 / all).ToString("F2").PadLeft(6)}% | " +
        $"Correct = {correct:N0}/{all:N0} | Time = {(sw.Elapsed.TotalMilliseconds / 1000.0).ToString("F3")}s");
    
    if (showFPS) 
        Console.WriteLine($"Test FPS = {10000 / sw.Elapsed.TotalSeconds:N0}");
}
// NETWORK AND DATA
struct Net
{
    public int[] net;
    public float[] weights;
    public int neuronLen;

    public Net(int[] net, int seed)
    {
        this.neuronLen = net.Sum();
        this.net = net;
        this.weights = Glorot(this.net, new(seed));
    }

    static float[] Glorot(int[] net, Random rnd)
    {
        int len = 0;
        for (int n = 0; n < net.Length - 1; n++)
            len += net[n] * net[n + 1];

        float[] weights = new float[len];

        for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i - 0] * net[i - 1]) // layer
        {
            float sd = MathF.Sqrt(6.0f / (net[i] + net[i + 1]));
            for (int m = w; m < w + net[i] * net[i + 1]; m++) // weights
                weights[m] = (float)rnd.NextDouble() * sd * 2 - sd; 
        }
        return weights;
    }
}
struct AutoData
{
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;

    static float[] NormalizeData(byte[] samples)
    {
        float[] samplesF = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            samplesF[i] = samples[i] / 255f;
        return samplesF;
    }

    public AutoData(string yourPath)
    {
        // Hardcoded URLs from my GitHub
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        byte[] test, training;

        // Change variable names for readability
        string trainDataPath = "trainData", trainLabelPath = "trainLabel", testDataPath = "testData", testLabelPath = "testLabel";

        if (!File.Exists(Path.Combine(yourPath, trainDataPath))
            || !File.Exists(Path.Combine(yourPath, trainLabelPath))
            || !File.Exists(Path.Combine(yourPath, testDataPath))
            || !File.Exists(Path.Combine(yourPath, testLabelPath)))
        {
            Console.WriteLine("Status: MNIST Dataset Not found");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // Padding bits: data = 16, labels = 8
            Console.WriteLine("Action: Downloading and Cleaning the Dataset from GitHub");
            training = new HttpClient().GetAsync(trainDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(60000 * 784).ToArray();
            labelsTraining = new HttpClient().GetAsync(trainLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(60000).ToArray();
            test = new HttpClient().GetAsync(testDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(10000 * 784).ToArray();
            labelsTest = new HttpClient().GetAsync(testLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(10000).ToArray();

            Console.WriteLine("Save Path: " + yourPath + "\n");
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainDataPath), training);
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainLabelPath), labelsTraining);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testDataPath), test);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testLabelPath), labelsTest);
        }
        else
        {
            // Data exists on the system, just load from yourPath
            Console.WriteLine("Dataset: MNIST (" + yourPath + ")" + "\n");
            training = File.ReadAllBytes(Path.Combine(yourPath, trainDataPath)).Take(60000 * 784).ToArray();
            labelsTraining = File.ReadAllBytes(Path.Combine(yourPath, trainLabelPath)).Take(60000).ToArray();
            test = File.ReadAllBytes(Path.Combine(yourPath, testDataPath)).Take(10000 * 784).ToArray();
            labelsTest = File.ReadAllBytes(Path.Combine(yourPath, testLabelPath)).Take(10000).ToArray();
        }

        samplesTrainingF = NormalizeData(training);
        samplesTestF = NormalizeData(test);
    }
}