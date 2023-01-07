#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "openacc.h"
#include <string_view>
#include <memory>
#include <unistd.h>
#include <stdio.h>
#include "network.hpp"
#include "mnist.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "ops.hpp" 
#include "utils.hpp"
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

using Neural::Tensor4D;
using Neural::Shape4D;
using Neural::LabeledData;
using Neural::Network;
using namespace std;

vector<Neural::LabeledData<double>> read_mnist_data() {
    // Load the data
    LOGI << "Reading mnist data new";
    Tensor4D<double> * original_data = read_mnist_images<double>("data/train-images-idx3-ubyte");
    
    LOGI << "Reading mnist labels";
    Tensor4D<int>* original_labels = read_mnist_labels("data/train-labels-idx1-ubyte");

    LOGI << "Splitting dataset";
    vector<LabeledData<double>> train_valid_test = split_dataset(original_data, original_labels, 0.2);

    LOGI << "Deleting orignal_data";
    delete original_data;

    LOGI << "Deleting orignal_labels";
    delete original_labels;

    LOGI << "Reading test_data, test_labels";
    LabeledData<double> test_data_labeled(read_mnist_images<double>("data/t10k-images-idx3-ubyte"), read_mnist_labels("data/t10k-labels-idx1-ubyte"));

    train_valid_test.push_back(test_data_labeled);

    return train_valid_test;
}

namespace plog
{
    class MyFormatter
    {
    public:
        static util::nstring header() { return "Header"; };
        static util::nstring format(const Record& record) {
            util::nstring ret = record.getMessage();

            ret = ret + "\n";
            return ret;
        };
    };
}

int main(int argc, char *argv[]) {
    printf("Hello World Classes training new\n");
    
    /*
        enum Severity {
            none = 0,
            fatal = 1,
            error = 2,
            warning = 3,
            info = 4,
            debug = 5,
            verbose = 6
        }
    */
    cout << "logging_level = atoi(argv[1])" << endl;
    string logging_level_str = argv[1];
    plog::Severity logging_level;

    if(logging_level_str == "fatal") logging_level = plog::fatal;
    else if(logging_level_str == "error") logging_level = plog::error;
    else if(logging_level_str == "warning") logging_level = plog::warning;
    else if(logging_level_str == "info") logging_level = plog::info;
    else if(logging_level_str == "debug") logging_level = plog::debug;
    else if(logging_level_str == "verbose") logging_level = plog::verbose;
    else if(logging_level_str == "none") logging_level = plog::none;
    else {
        throw(std::invalid_argument("Logging level invalid"));
    }
    cout << "Init plog" << endl;
    plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(logging_level, &consoleAppender ); // Initialize logging to the file.
    
    // // cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    // // cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;

    Tensor4D<double> *train_data, *valid_data, *test_data;
    Tensor4D<int> *train_labels, *valid_labels, *test_labels;

    vector<int> filter_size_conv1, filter_size_conv2, stride_conv1, stride_conv2;
    int depth_conv1, depth_conv2, num_hidden_nodes, num_outputs;
    string padding_conv1, padding_conv2;

    auto mnist_data = read_mnist_data();
    train_data = mnist_data[0].get_data();
    train_labels = mnist_data[0].get_labels();
    valid_data = mnist_data[1].get_data();
    valid_labels = mnist_data[1].get_labels();
    test_data = mnist_data[2].get_data();
    test_labels = mnist_data[2].get_labels();
    
    padding_conv1 = "same";
    padding_conv2 = "same";
    filter_size_conv1 = {5, 5};
    filter_size_conv2 = {5,5};
    stride_conv1 = {1,1};
    stride_conv2 = {1,1};
    depth_conv1 = 64;
    depth_conv2 = 64;
    num_hidden_nodes = 256;
    num_outputs = 10;

    Network testnet(train_data->shape()); //destructor?

    LOGI << "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv1 << ", \"relu\", " << filter_size_conv1[0] << ", " << stride_conv1[0] << ", \"" << padding_conv1 << "\")";
    testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, padding_conv1);
   
    LOGI << "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv2 << ", \"relu\", " << filter_size_conv2[0] << ", " << stride_conv2[0] << ", \"" << padding_conv2 << "\")";
    testnet.add_layer<Neural::Layers::Conv>(depth_conv2, "relu", filter_size_conv2, stride_conv2, padding_conv2);
    
    LOGI << "testnet.add_layer<Neural::Layers::Fc>(" << num_hidden_nodes << ", \"relu\")";
    testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");
    
    LOGI << "testnet.add_layer<Neural::Layers::Fc>(" << num_outputs << ", \"softmax\")";
    testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");

    int batch_size;
    LOGI << "batch_size = atoi(argv[2])";
    batch_size= atoi(argv[2]);
    assert(batch_size <= train_data->shape()[0]);
    LOGI << "batch_size = " << batch_size;

    double learning_rate = 0.05;
    
    LOGI << "testnet.train(train_data_tensor, train_labels_tensor, " << batch_size << ", true, " << learning_rate << ", \"CrossEntropy\")";
    testnet.train(train_data, train_labels, valid_data, valid_labels, batch_size, true, learning_rate, "CrossEntropy");

    return 0;
}
