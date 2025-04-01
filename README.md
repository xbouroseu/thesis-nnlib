# NNLib
## A C++ framework, inspired by TensorFlow/Keras, for training and evaluation of deep neural networks in Nvidia GPU-accelerated systems. 

### Example usage

Provides the `Neural` namespace plus `Neural::Network`, `Neural::Layers:Fc`, `Neural::Layers:Conv`, `Neural::Tensor4D`, `Neural::Shape4D` classes.

```cpp
#include "tensor.hpp"
#include "network.hpp"
#include "layer.hpp"

using Neural::Tensor4D;
using Neural::Shape4D;
using Neural::Network;

Tensor4D<double> train_data, valid_data, test_data; // assume initialized
Tensor4D<int> train_labels, valid_labels, test_labels; // assume initialized

// We will create a 3-layer network with a Conv->Fc->Output architecture.
// Initialize network with input data shape but with batch_size=undefined
// train_data.shape() := Shape4D(num_samples, channels, width, height)
// testnet.input_shape := Shape4D(-1, channels, width, height)
Network testnet(train_data.shape());

// Add Conv Layer with activation
int depth_conv1 = 64;
int filter_size_conv1 = 5; // can also be vector<int>(a,b)
int stride_conv1 = 1; // 1-stride in all directions. Can also be vector<int>(x,y) meaning x-stride horizontal, y-stride vertical
string padding_type_conv1 = "same"; // can also be "valid"

testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, padding_type_conv1);

// Add hidden FC layer with activation
int num_hidden_nodes = 256;
testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");

// Add output layer
int num_outputs = 10;
testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");

// Set hyperparameters
int batch_size = 32, max_epochs=0, max_steps_per_epoch=0; // 0=default, won't stop until algorithm decides
double learning_rate = 0.1f;
bool accelerated_run = true;

// Train network with train and validation datasets
testnet.train(train_data, train_labels, valid_data, valid_labels, batch_size, accelerated_run, learning_rate, "CrossEntropy", max_epochs, max_steps_per_epoch);

// Evaluate network against test dataset and obtain precision, recall, accuracy and f1_score metrics
double precision_test, recall_test, accuracy, f1_score;
testnet.eval(test_data, test_labels, recall_test, precision_test, accuracy, f1_score);
```

## Installation
### Requirements
- Nvidia GPU
- [Nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker

### Step 1: Pull the docker image
```
docker pull sirmihawk/thesis:hpc24.7_build
```

### Step 2: Create a new docker container
The `--rm` option will create a container which will be auto-removed once the session is ended.
```
docker run -it --rm --gpus all sirmihawk/thesis:hpc24.7_build
```

### Step 3: Build the library and sample apps
Clone the repository inside the container:
```
git clone https://github.com/xbouroseu/thesis-nnlib
cd thesis-nnlib
```

Build the framework and samples with:
```
make all
```

Alternatively you can only build the library with:
```
make lib
``` 

and the samples with:
```
make examples
```

### Step 4: Run the sample MNIST training application
After building the library and the sample apps we can run one example application which is training and evaluating a Convolution Neural Network on the MNIST dataset.


```
cd samples/mnist_app
```

The following command is for log-level `info` and batch_size `x`. We can also choose to run either the gpu-accelerated version or the non-accelerated one.

For the gpu-accelerated version:
```
./build/mnist_acc info x
```

For the non-accelerated version:
```
./build/mnist_noacc info x
```
