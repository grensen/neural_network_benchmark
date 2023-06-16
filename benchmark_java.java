import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {

        // define neural network
        int[] network = { 784, 100, 100, 10 };
        float LEARNINGRATE = 0.0005f;
        float MOMENTUM = 0.67f;
        int EPOCHS = 50;
        int BATCHSIZE = 800;
        float FACTOR = 0.999f;
        int SEED = 1337;
        boolean PARALLEL = true;

        printUserInfo();

        // Instantiate AutoData object
        AutoData data = new AutoData("C:\\mnistJava3\\");

        if(false) printImage(0, data);

        printHyperParameters(network, LEARNINGRATE, MOMENTUM, EPOCHS, BATCHSIZE, FACTOR, SEED);

        Net NN = new Net(network, SEED);

        runTraining(PARALLEL, data, NN, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

        runTest(PARALLEL, data, NN, 10000);

        System.out.println("\nBenchmark completed successfully");
    }
    private static void runTraining(boolean multiCore, AutoData d, Net NN, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE) {
        
        System.out.println("\nTraining Progress" + (multiCore ? " - Parallel" : "") + ":\n--------------------------------");

        float[] deltas = new float[NN.weights.length];

        long startTime = System.currentTimeMillis();

        for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR) {
            boolean[] c = new boolean[B * BATCHSIZE]; // for proper parallel count

            if (multiCore) {
                for (int b = 0; b < B; b++) {
                    final int batchStart = b * BATCHSIZE;
                    final int batchEnd = (b + 1) * BATCHSIZE;

                    IntStream.range(batchStart, batchEnd).parallel().forEach(x -> {
                        c[x] = train(Arrays.copyOfRange(
                                d.samplesTrainingF, x * 784, (x * 784) + 784),
                                d.labelsTraining[x]& 0xFF, NN, deltas);
                    });
                    sgd(0, NN.weights.length, NN.weights, deltas, lr, mom);
                }
            } else { // single core
                for (int b = 0; b < B; b++) {
                    final int batchStart = b * BATCHSIZE;
                    final int batchEnd = (b + 1) * BATCHSIZE;

                    for (int x = batchStart; x < batchEnd; x++) {
                        c[x] = train(Arrays.copyOfRange(
                                d.samplesTrainingF, x * 784, (x * 784) + 784),
                                d.labelsTraining[x]& 0xFF, NN, deltas);
                    }
                    sgd(0, NN.weights.length, NN.weights, deltas, lr, mom);
                }
            }

            if ((epoch + 1) % 10 == 0)
            { // info
                String epochInfo = String.format("Epoch = %3d |", epoch + 1);
                int count = countTrue(c);
                printInfo(epochInfo, count, B * BATCHSIZE, startTime, false);
            }
        }

    }
    private static boolean train(float[] sample, int target, Net NN, float[] deltas) {

        float[] neurons = new float[NN.neuronLen];

        int prediction = eval(sample, NN.net, NN.weights, neurons);

        softmax(neurons, neurons.length - NN.net[NN.net.length - 1]);

        if (neurons[NN.neuronLen - NN.net[NN.net.length - 1] + target] < 0.99)
            backprop(NN.net, NN.weights, neurons, deltas, target);

        return prediction == target;
    }
    private static void runTest(boolean multiCore, AutoData d, Net NN, int len) {
        
        System.out.println("\nTest Results" + (multiCore ? " - Parallel" : "") + ":\n--------------------------------");

        boolean[] c = new boolean[len]; // for proper parallel count

        long startTime = System.currentTimeMillis();

        if (multiCore)
        {
            IntStream.range(0, len).parallel().forEach(x -> {
                c[x] = test(Arrays.copyOfRange(d.samplesTestF, x * 784, (x * 784) + 784),
                        d.labelsTest[x] & 0xFF, NN);
            });
        } else { // single core
            for (int x = 0; x < len; x++) {
                c[x] = test(Arrays.copyOfRange(d.samplesTestF, x * 784, (x * 784) + 784),
                        d.labelsTest[x] & 0xFF, NN);
            }
        }

        printInfo("Test", countTrue(c), len, startTime, true);
    }
    private static boolean test(float[] sample, int target, Net NN) {
        
        int prediction = eval(sample, NN.net, NN.weights, new float[NN.neuronLen]);
        
        return prediction == target;
    }
    private static int eval(float[] sample, int[] net, float[] weights, float[] neurons) {
        
        System.arraycopy(sample, 0, neurons, 0, 784);

        feedForward(net, weights, neurons);

        return argmax(Arrays.copyOfRange(neurons, neurons.length - net[net.length - 1], neurons.length));
    }
    static void feedForward(int[] net, float[] weights, float[] neuron) {
        
        for (int i = 0, k = net[0], m = 0, j = 0; i < net.length - 1; i++) {
            
            int left = net[i], right = net[i + 1];
            for (int l = 0, w = m; l < left; l++, w += right) {
                float n = neuron[j + l];
                if (n > 0)
                    for (int r = 0; r < right; r++)
                        neuron[k + r] += n * weights[w + r];
            }
            m += left * right; j += left; k += right;
        }
    }
    private static int argmax(float[] neurons) {
        
        int id = 0;
        float max = neurons[0];
        for (int i = 1; i < neurons.length; i++) {
            float n = neurons[i];
            if (n > max) {
                id = i;
                max = n;
            }
        }
        return id; // prediction
    }
    private static void softmax(float[] neurons, int start) {
        
        float scale = 0;
        for (int n = start; n < neurons.length; n++) {
            neurons[n] = (float) Math.exp(neurons[n]); // activation then sum up
            scale += neurons[n];
        }
        scale = 1 / scale; // turns division to multiplication
        for (int n = start; n < neurons.length; n++) {
            neurons[n] *= scale; // probabilities
        }
    }
    private static void backprop(int[] net, float[] weights, float[] neuron, float[] delta, int target) {
        
        float[] gradient = new float[neuron.length];

        // target - output
        for (int r = neuron.length - net[net.length - 1], p = 0; r < neuron.length; r++, p++) {
            gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];
        }

        for (int i = net.length - 2, j = neuron.length - net[net.length - 1], k = neuron.length, m = weights.length; i >= 0; i--) {
            int left = net[i], right = net[i + 1];
            m -= right * left; j -= left; k -= right;

            for (int l = 0, w = m; l < left; l++) {
                float n = neuron[j + l];
                float sum = 0;
                if (n > 0) {
                    for (int r = 0; r < right; r++) {
                        float g = gradient[k + r];
                        delta[w + r] += n * g;
                        sum += weights[w + r] * g;
                    }
                    gradient[j + l] = sum;
                }
                w += right;
            }
        }
    }
    private static void sgd(int start, int end, float[] weights, float[] delta, float lr, float mom) {
        
        for (int w = start; w < end; w++) {
            weights[w] = delta[w] * lr + weights[w];
            delta[w] *= mom;
        }
    }
    private static void printImage(int id, AutoData data) {
        float[] firstImage = Arrays.copyOfRange(data.samplesTrainingF, id * 784, (id + 1) * 784);
        // Print the image
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                float pixelValue = firstImage[i * 28 + j];
                if (pixelValue > 0.0) {
                    System.out.print("# ");   // Print a character for a bright pixel
                } else {
                    System.out.print("  ");   // Print a space for a dark pixel
                }
            }
            System.out.println();  // Move to the next line
        }
        System.out.println("Label: " + data.labelsTraining[id] + "\n");
    }
    private static int countTrue(boolean[] array) {
        int count = 0;
        for (boolean value : array) {
            if (value) {
                count++;
            }
        }
        return count;
    }
    private static void printInfo(String str, int correct, int all, long startTime, boolean showFPS) {
        double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0;
        System.out.println(str + " Accuracy = " + String.format("%.2f", (correct * 100.0 / all)) + "% | " +
                "Correct = " + correct + "/" + all + " | Time = " + String.format("%.3f", elapsedTime) + "s");

        if (showFPS)
            System.out.println("Test FPS = " + String.format("%.0f", 10000 / elapsedTime));
    }
    private static void printUserInfo() {
        System.out.println("\nNeural Network Benchmark - " + LocalDate.now().getYear() + " | Java\n");
        System.out.println("CPU: AMD Ryzen 5 5600X | Cores: 6 | Default Speed: 3.7 GHz\n"); // Change it according to your system
    }
    private static void printHyperParameters(int[] network, float learningRate, float momentum, int epochs, int batchSize, float factor, int seed) {
        System.out.println("Neural Network Configuration:");
        System.out.println("NETWORK      = " + Arrays.toString(network));
        int weightsSum = 0;
        for (int i = 0; i < network.length - 1; i++) {
            weightsSum += network[i] * network[i + 1];
        }
        System.out.println("WEIGHTS      = " + weightsSum);
        System.out.println("SEED         = " + seed);
        System.out.println("LEARNINGRATE = " + String.format("%.4f", learningRate));
        System.out.println("MOMENTUM     = " + momentum);
        System.out.println("BATCHSIZE    = " + batchSize);
        System.out.println("EPOCHS       = " + epochs);
        System.out.println("FACTOR       = " + factor);
    }
}

class Net {
    public int[] net;
    public float[] weights;
    public int neuronLen;
    public Net(int[] net, int seed) {
        this.neuronLen = Arrays.stream(net).sum();
        this.net = net;
        this.weights = glorot(this.net, new Random(seed));
    }
    private float[] glorot(int[] net, Random rnd) {
        int len = 0;
        for (int n = 0; n < net.length - 1; n++)
            len += net[n] * net[n + 1];

        float[] weights = new float[len];

        for (int i = 0, w = 0; i < net.length - 1; i++, w += net[i - 0] * net[i - 1]) { // layer
            float sd = (float)Math.sqrt(6.0 / (net[i] + net[i + 1]));
            for (int m = w; m < w + net[i] * net[i + 1]; m++) { // weights
                weights[m] = (float)( rnd.nextDouble() * sd * 2 - sd);
            }
        }
        return weights;
    }
}

class AutoData {
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;
    public AutoData(String yourPath) {
        // Hardcoded URLs from my GitHub
        String trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        String trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        String testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        String testLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        byte[] test, training;

        // Change variable names for readability
        String trainDataPath = "trainData", trainLabelPath = "trainLabel", testDataPath = "testData", testLabelPath = "testLabel";

        if (!Files.exists(Path.of(yourPath, trainDataPath))
                || !Files.exists(Path.of(yourPath, trainLabelPath))
                || !Files.exists(Path.of(yourPath, testDataPath))
                || !Files.exists(Path.of(yourPath, testLabelPath))) {
            System.out.println("Status: MNIST Dataset Not found");
            if (!Files.exists(Path.of(yourPath))) {
                try {
                    Files.createDirectory(Path.of(yourPath));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            // Padding bits: data = 16, labels = 8
            System.out.println("Action: Downloading and Cleaning the Dataset from GitHub");
            training = downloadData(trainDataUrl, yourPath, trainDataPath, 16, 60000 * 784);
            labelsTraining = downloadData(trainLabelUrl, yourPath, trainLabelPath, 8, 60000);
            test = downloadData(testDataUrl, yourPath, testDataPath, 16, 10000 * 784);
            labelsTest = downloadData(testLabelUrl, yourPath, testLabelPath, 8, 10000);

            System.out.println("Save Path: " + yourPath + "\n");
        } else {
            // Data exists on the system, just load from yourPath
            System.out.println("Dataset: MNIST (" + yourPath + ")" + "\n");
            try {
                training = Arrays.copyOfRange(Files.readAllBytes(Path.of(yourPath, trainDataPath)), 16, 60000 * 784 + 16);
                labelsTraining = Arrays.copyOfRange(Files.readAllBytes(Path.of(yourPath, trainLabelPath)), 8, 60000 + 8);
                test = Arrays.copyOfRange(Files.readAllBytes(Path.of(yourPath, testDataPath)), 16, 10000 * 784 + 16);
                labelsTest = Arrays.copyOfRange(Files.readAllBytes(Path.of(yourPath, testLabelPath)), 8, 10000 + 8);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }
        }

        samplesTrainingF = normalizeData(training);
        samplesTestF = normalizeData(test);
    }
    private static float[] normalizeData(byte[] samples) {
        float[] samplesF = new float[samples.length];
        for (int i = 0; i < samples.length; i++)
            samplesF[i] = (samples[i] & 0xFF) / 255f; // Convert byte to unsigned int
        return samplesF;
    }
    private static byte[] downloadData(String url, String savePath, String fileName, int skipBytes, int takeBytes) {
        try {
            URL downloadUrl = new URL(url);
            HttpURLConnection connection = (HttpURLConnection) downloadUrl.openConnection();
            connection.setRequestMethod("GET");

            InputStream inputStream = connection.getInputStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            OutputStream outputStream = new FileOutputStream(savePath + File.separator + fileName);

            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }

            outputStream.close();
            inputStream.close();

            return Arrays.copyOfRange(Files.readAllBytes(Path.of(savePath, fileName)), skipBytes, skipBytes + takeBytes);

        } catch (IOException e) {
            e.printStackTrace();
        }

        return new byte[0];
    }
}