#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <string_view>
#include <memory>
#include <unistd.h>
#include <stdio.h>
#include "utils.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "layer.hpp"
#include "mnist.hpp"
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

//cm here
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

    LOGI << "Spliting dataset";
    vector<LabeledData<double>> train_valid_test = split_dataset(original_data, original_labels, 0.2);

    LOGI << "Deleting original_data";
    delete original_data;

    LOGI << "Deleting original_labels";
    delete original_labels;

    LOGI << "Reading test_data, test_labels";
    LabeledData<double> test_data_labeled(read_mnist_images<double>("data/t10k-images-idx3-ubyte"), read_mnist_labels("data/t10k-labels-idx1-ubyte"));

    train_valid_test.push_back(test_data_labeled);

    return train_valid_test;
}

int main(int argc, char *argv[]) {
    printf("Hello World Classes training all while new\n");
    cout << "__FILE__ = " << __FILE__ << endl;
    cout << "logging_level = atoi(argv[1])" << endl;
    string logging_level_str = argv[1];
    plog::Severity logging_level;

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
    plog::ColorConsoleAppender<plog::MyFormatter> consoleAppender;
    plog::init(logging_level, &consoleAppender ); // Initialize logging to the file.
    
    LOGI << "Neural::get_device_type(gpu=4, host=2): " << Neural::get_device_type();

    // // cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    // // cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;

    unique_ptr<Tensor4D<double>> train_data, valid_data, test_data;
    unique_ptr<Tensor4D<int>> train_labels, valid_labels, test_labels;

    vector<int> filter_size_conv1, filter_size_conv2, stride_conv1, stride_conv2;
    int depth_conv1, depth_conv2, num_hidden_nodes, num_outputs;
    string padding_conv1, padding_conv2;

    PLOGI << "calling read_mnist_data()";
    auto mnist_data = read_mnist_data();
    train_data.reset(mnist_data[0].get_data());
    train_labels.reset(mnist_data[0].get_labels());
    valid_data.reset(mnist_data[1].get_data());
    valid_labels.reset(mnist_data[1].get_labels());
    test_data.reset(mnist_data[2].get_data());
    test_labels.reset(mnist_data[2].get_labels());
    
    Shape4D train_data_shape = train_data->shape();
    int B = train_data_shape[0], C = train_data_shape[1], H = train_data_shape[2], W = train_data_shape[3];

    double *train_data_data = train_data->data();
    PLOGD << "train_data[1]";
    for(int b = 0; b < 1; b++) {
        for(int c = 0; c < C; c++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    PLOGD << train_data_data[b*C*H*W + c *H*W + h*W + w];
                }
            }
        }
    }
    PLOGD << "/train_data[1]";
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
//TODO find solution to data locality, relative? cmd argument ?, work only by running inside app folder?

    Network testnet(train_data->shape()); //destructor?

    PLOGI << "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv1 << ", \"relu\", " << filter_size_conv1[0] << ", " << stride_conv1[0] << ", \"" << padding_conv1 << "\")";
    testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, padding_conv1);
   
    PLOGI << "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv2 << ", \"relu\", " << filter_size_conv2[0] << ", " << stride_conv2[0] << ", \"" << padding_conv2 << "\")";
    testnet.add_layer<Neural::Layers::Conv>(depth_conv2, "relu", filter_size_conv2, stride_conv2, padding_conv2);
    
    PLOGI << "testnet.add_layer<Neural::Layers::Fc>(" << num_hidden_nodes << ", \"relu\")";
    testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");
    
    PLOGI << "testnet.add_layer<Neural::Layers::Fc>(" << num_outputs << ", \"softmax\")";
    testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");

    int batch_size;
    PLOGI << "batch_size = atoi(argv[2])";
    batch_size= atoi(argv[2]);
    assert(batch_size <= train_data->shape()[0]);
    PLOGI << "batch_size = " << batch_size;

    int fsteps=0, fepochs=0;

    if(argc>=4) { fepochs=atoi(argv[3]); }
    if(argc>=5) { fsteps=atoi(argv[4]); }

    double learning_rate = 0.05;
    
    LOGW.printf("testnet.train(*train_data.get(), *train_labels.get(), *valid_data.get(), *valid_labels.get(), %d, true, %f, %s, %d, %d)",batch_size, learning_rate, "CrossEntropy", fepochs, fsteps);
    testnet.train(*train_data.get(), *train_labels.get(), *valid_data.get(), *valid_labels.get(), batch_size, true, learning_rate, "CrossEntropy", fepochs, fsteps);

    double precision_test, recall_test, accuracy_test, f1_score_test;
    LOGW << "testnet.eval(*test_data.get(), *test_labels.get(),recall_test, precision_test, accuracy_test, f1_score_test)";
    testnet.eval(*test_data.get(), *test_labels.get(),recall_test, precision_test, accuracy_test, f1_score_test);
    LOGW << endl << endl;
    LOGW << "Precision: " << precision_test << " | Recall: " << recall_test << " | Accuracy: " << accuracy_test << " | F1_score: " << f1_score_test;
    LOGW << endl << endl;
    return 0;
}

