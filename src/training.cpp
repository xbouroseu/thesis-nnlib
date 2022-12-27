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

int main(int argc, char *argv[]) {
    printf("Hello World Classes training\n");
    printf("Current working dir: %s\n", get_current_dir_name());
    cout << "__FILE__" << __FILE__ << endl;

    // cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    // cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;
    LOG("THIS IS LOGGED");
    cout << "Neural::is_acc " << Neural::is_acc << endl;
    cout << "Neural::get_device_type() " << Neural::get_device_type() << endl;
    
    /////////////////////////////////////////
//     cout << "Init test_input" << endl;
//     
//     int num_s = 5, num_ch = 3, num_r = 3, num_c = 3;
//     double *test_input = new double[num_s*num_ch*num_r*num_c];
//     
//     for(int s = 0; s < num_s; s++) {
//         for(int ch=0; ch<num_ch; ch++) {
//             for(int i = 0; i < num_r; i++) {
//                 for(int j = 0; j < num_c; j++) {
//                     int val = s*num_ch*num_r*num_c + ch*num_r*num_c + i*num_c + j + 1;
//                     int vl = 31*val;
//                     int vll = vl%255;
//                     test_input[s*num_ch*num_r*num_c + ch*num_r*num_c + i*num_c + j] = vll;            
//                 }
//             }
// 
//         }
//     }
//     
//     vector<vector<cv::Mat>> mdata;
//     for(int s = 0; s < num_s; s++) {
//         vector<cv::Mat> sdata;
//         for(int ch=0; ch<num_ch; ch++) {
//             sdata.push_back(cv::Mat(num_r, num_c, CV_64FC1, test_input+s*num_ch*num_r*num_c + ch*num_r*num_c));
//         }
//         mdata.push_back(sdata);
//     }
//     for(int i = 0; i < mdata.size(); i++) {
//         for(int s = 0; s < mdata[i].size(); s++) {
//             cout << mdata[i][s] << endl;
//         }
//     }
//     
//     int num_out = 5;
//     
//     int *test_labels = new int[num_s*num_out];
//     
//     cout << "Init test labels" << endl;
//     for(int s = 0; s < num_s; s++) {
//         int lbl = s%num_out;
//         
//         for(int ot = 0; ot < num_out; ot++) {
//             if(lbl == ot) {
//                 test_labels[s*num_out + ot] = 1;
//             }
//             else {
//                 test_labels[s*num_out + ot] = 0;
//             }
//         }
//     }
// 
//     cout << "Convert to cv::Mat" << endl;
//     cv::Mat mdata_labels(num_s, num_out, CV_32S, test_labels);
//     
//     cout << "Print labels" << endl;
//     cout << mdata_labels << endl;
//         
//     ////////////////////////////////////////////
//     
//     cout << "Init ltest_data" << endl;
//     Tensor4D<double> * ltest_data = new Tensor4D<double>(test_input, Shape4D(num_s, num_ch, num_r, num_c));
//     ltest_data->print();
//     
//     cout << "Init ltest_labels" << endl;
//     Tensor4D<int> * ltest_labels = new Tensor4D<int>(test_labels, Shape4D(num_s, num_out));
//     ltest_labels->print();    
//     
    // int bs = 2;
    
/*    terminal.integrated.defaultLocation
    cout << "Uni.get: " << uni.get() << " | uni == nullptr: " << (uni.get()==nullptr) << endl;
    
    uni.reset();
    cout << "Uni.get: " << uni.get() << " | uni == nullptr: " << (uni.get()==nullptr) << endl;*/


    vector<LabeledData<double>> mnist_data = read_data();
    LabeledData<double> train_data = mnist_data[0];
    LabeledData<double> valid_data = mnist_data[1];
    LabeledData<double> test_data = mnist_data[2];

    LOG("Init network");
    Network *testnet = new Network(train_data.data->shape()); //destructor?
    
    vector<int> filter_size_conv1 = {5, 5}, filter_size_conv2 = {5,5}, stride_conv1 = {1,1}, stride_conv2 = {1,1};
    int depth_conv1 = 64, depth_conv2 = 64, num_hidden_nodes = 256, num_outputs = 10;

    LOG("testnet.add_layer<Neural::Layers::Conv>(" << depth_conv1 << ", \"relu\", " << filter_size_conv1[0] << ", " << stride_conv1[0] << ", \"same\")");
    testnet->add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, "same");
   
    LOG("testnet.add_layer<Neural::Layers::Conv>(" << depth_conv2 << ", \"relu\", " << filter_size_conv2[0] << ", " << stride_conv2[0] << ", \"same\")");
    testnet->add_layer<Neural::Layers::Conv>(depth_conv2, "relu", filter_size_conv2, stride_conv2, "same");
    
    LOG("testnet.add_layer<Neural::Layers::Fc>(" << num_hidden_nodes << ", \"relu\")");
    testnet->add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");
    
    LOG("testnet.add_layer<Neural::Layers::Fc>(" << num_outputs << ", \"softmax\")");
    testnet->add_layer<Neural::Layers::Fc>(num_outputs, "softmax");
    
    LOG("steps = atoi(argv[1])");
    int steps = atoi(argv[1]);
    LOG("Steps = " << steps);

    LOG("batch_size = atoi(argv[2])");
    int batch_size= atoi(argv[2]);
    LOG("batch_size = " << batch_size);
    double learning_rate = 0.05;
    
    LOG("testnet.train(train_data_tensor, train_labels_tensor, " << batch_size << ", " << steps << ", true, " << learning_rate << ", \"CrossEntropy\")");
    testnet->train(train_data, valid_data, batch_size, steps, true, learning_rate, "CrossEntropy");
    
    delete train_data.data;
    delete train_data.labels;
    delete valid_data.data;
    delete valid_data.labels;
    delete test_data.data;
    delete test_data.labels;
    // delete ltest_data;
    // delete ltest_labels;
    // delete[] test_input;
    // delete[] test_labels;
    delete testnet;
    return 0;
}
