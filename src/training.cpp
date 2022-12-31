#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "openacc.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <curand.h> 
#include <string_view>
#include <memory>
#include <unistd.h>
#include <stdio.h>
// #include <math.h>
//#include <cuchar>
//#include <random>
//#include <curand_kernel.h>

#include "network.hpp"
#include "mnist.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "ops.hpp" 
#include "utils.hpp"

using Neural::Tensor4D;
using Neural::Shape4D;
using Neural::LabeledData;
using Neural::Network;
using namespace std;
//using namespace cv;

template <typename T> 
constexpr auto type_name() {
    std::string_view name, prefix, suffix;
    
    #ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
    #elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
    #elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
    #endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    
    return name;
}

vector<Neural::LabeledData<double>> read_data() {
    // Load the data
    cout << "Reading mnist data transformed" << endl;
    Tensor4D<double> * original_data = read_mnist_images<double>("data/train-images-idx3-ubyte");
    Tensor4D<int>* original_labels = read_mnist_labels("data/train-labels-idx1-ubyte");

    vector<LabeledData<double>> train_valid_test = split_dataset(original_data, original_labels, 0.2);

    delete original_data;
    delete original_labels;
    LabeledData<double> test_data_labeled;
    
    test_data_labeled.data = read_mnist_images<double>("data/t10k-images-idx3-ubyte");
    test_data_labeled.labels = read_mnist_labels("data/t10k-labels-idx1-ubyte");

    train_valid_test.push_back(test_data_labeled);
    LOG("/read_data");

    return train_valid_test;
}

Tensor4D<double> * read_sample_data(string data_path) {
    ifstream file_data(data_path);

    if(file_data.is_open()) {
        int magic_n, ns, sh, sw;

        file_data >> magic_n >> ns >> sh >> sw;

        if(magic_n != 3344) {
            throw runtime_error("Invalid SAMPLE data file!");
        }

        printf("Magic number: %d, num_samples: %d, sample_height: %d, sample_width: %d\n", magic_n, ns, sh, sw);
        
        Tensor4D<double> * _dataset = new Tensor4D<double>(ns, 1, sh, sw);

        for(int i = 0; i < _dataset->size(); i++) {
            file_data >> _dataset->iat(i);
        }

        file_data.close();

        return _dataset;
    }
    else {
        throw runtime_error("Unable to open file " + data_path + "!");
    }

}


Tensor4D<int> * read_sample_labels(string data_path) {
    ifstream file_data(data_path);

    if(file_data.is_open()) {
        int magic_n, ns;

        file_data >> magic_n >> ns;

        if(magic_n != 3345) {
            throw runtime_error("Invalid SAMPLE labels file!");
        }

        printf("Magic number: %d, num_labels\n", magic_n, ns);
        
        Tensor4D<int> * _dataset = new Tensor4D<int>(ns, 10, 1, 1);

        for(int i = 0; i < ns; i++) {
            int lblfull;
            file_data >> lblfull;
            for(int m = 0; m < 10; m++) {
                int lbl1hot=0;
                if(m == lblfull) {
                    lbl1hot = 1;
                }
                _dataset->iat(i*10 + m) = lbl1hot;
            }
        }

        file_data.close();

        return _dataset;
    }
    else {
        throw runtime_error("Unable to open file " + data_path + "!");
    }
}

int main(int argc, char *argv[]) {
    printf("Hello World Classes training\n");
    printf("Current working dir: %s\n", get_current_dir_name());
    cout << "__FILE__" << __FILE__ << endl;

    // cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    // cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;
    LOG("THIS IS LOGGED");
    cout << "Neural::is_acc " << Neural::is_acc << endl;
    cout << "Neural::get_device_type() " << Neural::get_device_type() << endl;
    
    unique_ptr<Tensor4D<double>> sample_data(read_sample_data("data/sample_data.txt"));
    unique_ptr<Tensor4D<int>> sample_labels(read_sample_labels("data/sample_labels.txt"));

    LabeledData<double> sample_labeled_data;
    sample_labeled_data.data = sample_data.get();
    sample_labeled_data.labels = sample_labels.get();

    // sample_data->print();
    // sample_labels->print();

    // vector<int> filter_size_conv1 = {3, 3}, filter_size_conv2 = {3,3}, stride_conv1 = {1,1}, stride_conv2 = {1,1};
    // int depth_conv1 = 4, depth_conv2 = 8, num_hidden_nodes = 4, num_outputs = 10;

    vector<LabeledData<double>> mnist_data = read_data();
    Network testnet(mnist_data[0].data->shape()); //destructor?
    
    vector<int> filter_size_conv1 = {5, 5}, filter_size_conv2 = {5,5}, stride_conv1 = {1,1}, stride_conv2 = {1,1};
    int depth_conv1 = 64, depth_conv2 = 64, num_hidden_nodes = 256, num_outputs = 10;

    LOG("testnet.add_layer<Neural::Layers::Conv>(" << depth_conv1 << ", \"relu\", " << filter_size_conv1[0] << ", " << stride_conv1[0] << ", \"same\")");
    testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, "same");
   
    LOG("testnet.add_layer<Neural::Layers::Conv>(" << depth_conv2 << ", \"relu\", " << filter_size_conv2[0] << ", " << stride_conv2[0] << ", \"same\")");
    testnet.add_layer<Neural::Layers::Conv>(depth_conv2, "relu", filter_size_conv2, stride_conv2, "same");
    
    LOG("testnet.add_layer<Neural::Layers::Fc>(" << num_hidden_nodes << ", \"relu\")");
    testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");
    
    LOG("testnet.add_layer<Neural::Layers::Fc>(" << num_outputs << ", \"softmax\")");
    testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");

    LOG("batch_size = atoi(argv[1])");
    int batch_size= atoi(argv[1]);
    LOG("batch_size = " << batch_size);

    double learning_rate = 0.05;
    
    LOG("testnet.train(train_data_tensor, train_labels_tensor, " << batch_size << ", true, " << learning_rate << ", \"CrossEntropy\")");
    testnet.train(mnist_data[0], mnist_data[1], batch_size, true, learning_rate, "CrossEntropy");
    
    // delete train_data.data;
    // delete train_data.labels;
    // delete valid_data.data;
    // delete valid_data.labels;
    // delete test_data.data;
    // delete test_data.labels;

    return 0;
}
