# thesis-nnlib
## A framework for building neural networks in nvidia-gpu accelerated systems

# Example Usage

```cpp
unique_ptr<Tensor4D<double>> train_data, valid_data, test_data; // placeholder. assume initialized
unique_ptr<Tensor4D<int>> train_labels, valid_labels, test_labels; // placeholder. assume initialized

// Initialize network with input data shape (channels, width, height). Batch size is left undefined.
Network testnet(train_data->shape());

// Add Conv Layer with activation
testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, padding_conv1);

// Add hidden FC layer with activation
testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");

// Add output layer
testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");

// Set hyperparameters
int batch_size, max_epochs, max_steps_per_epoch;
double learning_rate;
bool accelerated;

// Train network with train and validation datasets
testnet.train(*train_data.get(), *train_labels.get(), *valid_data.get(), *valid_labels.get(), batch_size, accelerated, learning_rate, "CrossEntropy", max_epochs, max_steps_per_epoch);

// Evaluate network against test dataset and obtain precision and recall metrics
double precision_test, recall_test;
testnet.eval(*test_data.get(), *test_labels.get(), recall_test, precision_test);
```

# Installation
## Requirements: NVIDIA GPU, Docker

## Step 1: Pull the docker image
```
$ docker pull sirmihawk/thesis:hpc22.11_build
```

## Step 2: Create a new docker container
The --rm option will create a container which will be auto-removed once the session is ended.

```
$ docker run -it --rm --gpus all sirmihawk/thesis:hpc22.11_build
```

## Step 3: Build the library and sample apps
Once you're inside the container you can clone again the repository with:
```
$ git clone https://github.com/xbouroseu/thesis-nnlib && cd thesis-nnlib
```
And then to build the library and apps you do:
```
$ make all
```

Alternatively you can only build the library with:
```
$ make lib
``` 

and the apps with:
```
$ make app
```

## Step 4: Run the MNIST training application
After building the library and the sample apps we can run one application which is training and evaluating a Convolution Neural Network on the MNIST dataset.

Let's assume the variable `x` is the batch size:

```
$ cd apps/mnist_app
```

To run the gpu-accelerated version:
```
$ ./build/mnist_acc info x
```

And the non-accelerated one:
```
$ ./build/mnist_noacc info x
```
