#include <cstdio>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include "utils.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "layer.hpp"

using Neural::Tensor4D;
using Neural::Network;

using namespace std;


Tensor4D<double> * read_sample_data(string data_path) {
    ifstream file_data(data_path);

    if(file_data.is_open()) {
        int magic_n, ns, sh, sw;

        file_data >> magic_n >> ns >> sh >> sw;

        if(magic_n != 3344) {
            throw runtime_error("Invalid SAMPLE data file!");
        }
        
        LOGI.printf("Magic number: %d, num_samples: %d, sample_height: %d, sample_width: %d\n", magic_n, ns, sh, sw);

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

        LOGI.printf("Magic number: %d, num_labels\n", magic_n, ns);

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
    cout << "logging_level = atoi(argv[1])" << endl;
    string logging_level_str = argv[1];
    plog::Severity logging_level;
    get_current_dir_name();
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

    Tensor4D<double> *train_data, *valid_data, *test_data;
    Tensor4D<int> *train_labels, *valid_labels, *test_labels;

    vector<int> filter_size_conv1, filter_size_conv2, stride_conv1, stride_conv2;
    int depth_conv1, depth_conv2, num_hidden_nodes, num_outputs;
    string padding_conv1, padding_conv2;
    
    train_data = read_sample_data("../data/sample_data_custom.txt");
    train_labels = read_sample_labels("../data/sample_labels.txt");
    valid_data = train_data;
    valid_labels = train_labels;
    
    test_data = train_data;
    test_labels = train_labels;

    _LLOG(info, train_data);
    _LLOG(info, train_labels);
    
    padding_conv1 = "valid";
    padding_conv2 = "valid";
    filter_size_conv1 = {2,2};
    filter_size_conv2 = {2,2};
    stride_conv1 = {1,1};
    stride_conv2 = {1,1};
    depth_conv1 = 2;
    depth_conv2 = 2;
    num_hidden_nodes = 5;
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

    int fsteps=0, fepochs=0;

    if(argc>=4) { fepochs=atoi(argv[3]); }
    if(argc>=5) { fsteps=atoi(argv[4]); }

    double learning_rate = 0.05;
    
    LOGW.printf("testnet.train(train_data_tensor, train_labels_tensor, %d, true, %f, %s, %d, %d)",batch_size, learning_rate, "CrossEntropy", fepochs, fsteps);

    testnet.train(train_data, train_labels, valid_data, valid_labels, batch_size, true, learning_rate, "CrossEntropy", fepochs, fsteps);
    
}   