s#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
//#include <cuchar>

//#include <random>
//#include <curand_kernel.h>
//#include <math.h>
//using namespace cv;
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <curand.h> 
#include <string_view>
#include <memory>
#include "openacc.h"
#include "neural.hpp"
#include "mnist.hpp"
#include "network.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "ops.hpp" 

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

using namespace std;

auto read_data() 

    // Load the data
    cout << "Reading mnist data transformed" << endl;
    Tensor4D<double>* train_img_data = read_mnist_images<double>("data/train-images-idx3-ubyte");
    Tensor4D<int>* train_labels_data = read_mnist_labels<double>("data/train-labels-idx1-ubyte");
    
    auto data_splited = split_data(*train_img_data, *train_labels_data, 0.2);
    delete train_img_data;
    delete train_labels_data;
    
    LabeledData<double> test_data_labeled;
    
    test_data_labeled.data = read_mnist_images<double>("data/train-images-idx3-ubyte");
    test_data_labeled.labels = read_mnist_labels<int>("data/train-labels-idx1-ubyte");

    
    data_splited.push_back(test_data_labeled);

    // TODO backprop divide by batch size, emltp? original
        
    return data_splited;
}

int main(int argc, char *argv[]) {
    printf("Hello World Classes training\n");
    // cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    // cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;
    
    cout << "Neural::is_acc " << Neural::is_acc << endl;
    cout << "Neural::get_device_type() " << Neural::get_device_type() << endl;
    
    auto mnist_data = read_data();
    
    auto train_data = mnist_data[0];
    auto valid_data = mnist_data[1];
    auto test_data = mnist_data[2];
    
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
    
/*    
    cout << "Uni.get: " << uni.get() << " | uni == nullptr: " << (uni.get()==nullptr) << endl;
    
    uni.reset();
    cout << "Uni.get: " << uni.get() << " | uni == nullptr: " << (uni.get()==nullptr) << endl;*/

    using Neural::Network;
    using Neural::Layers::Layer;
    cout<< "Init network" << endl;
    Network testnet; //destructor?
    
    int num_outputs = 10;
    int num_hidden_nodes = 256;
    int filter_size_conv1 = 5, depth_conv1 = 64, stride_conv1 = 1;
    int filter_size_conv2 = 5, depth_conv2 = 64, stride_conv2 = 1;
    
    cout<< "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv1 << ", \"relu\", " << filter_size_conv1 << ", " << stride_conv1 << ", \"same\")" << endl;
    testnet.add_layer<Neural::Layers::Conv>(depth_conv1, "relu", filter_size_conv1, stride_conv1, "same");
   
    cout<< "testnet.add_layer<Neural::Layers::Conv>(" << depth_conv2 << ", \"relu\", " << filter_size_conv2 << ", " << stride_conv2 << ", \"same\")" << endl;
    testnet.add_layer<Neural::Layers::Conv>(depth_conv2, "relu", filter_size_conv2, stride_conv2, "same");
    
    cout << "testnet.add_layer<Neural::Layers::Fc>(" << num_hidden_nodes << ", \"relu\")" << endl;
    testnet.add_layer<Neural::Layers::Fc>(num_hidden_nodes, "relu");
    
    // testnet.add_layer(new FcLayer(2, "relu"));
    cout << "testnet.add_layer<Neural::Layers::Fc>(" << num_outputs << ", \"softmax\")" << endl;
    testnet.add_layer<Neural::Layers::Fc>(num_outputs, "softmax");
    
    cout << "steps = atoi(argv[1])" << endl;
    int steps = atoi(argv[1]);
    cout << "Steps = " << steps << endl;

    cout << "batch_size = atoi(argv[2])" << endl;
    int batch_size= atoi(argv[2]);
    cout << "batch_size = " << batch_size << endl;
    double learning_rate = 0.05;
    cout << "testnet.train(train_data_tensor, train_labels_tensor, " << batch_size << ", " << steps << ", true, " << learning_rate << ", \"CrossEntropy\")" << endl;
    testnet.train(train_data, valid_data, batch_size, steps, true, learning_rate, "CrossEntropy");
    
    // delete ltest_data;
    // delete ltest_labels;
    // delete[] test_input;
    // delete[] test_labels;
    return 0;
}
