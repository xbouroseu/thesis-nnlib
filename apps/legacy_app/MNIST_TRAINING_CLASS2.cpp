#include <cstdio>
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
#include <memory>
#include "neural.hpp"
#include "openacc.h"
#include "mnist.hpp"
#include "network.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "ops.hpp"
#include <string_view>

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

int main(int argc, char *argv[]) {
    printf("Hello World Classes training\n");
    cout << type_name<decltype(std::function{acc_deviceptr})>() << endl;
    cout << type_name<decltype(std::function{Neural::deviceptr})>() << endl;
    
    cout << "Neural::is_acc " << Neural::is_acc << endl;
    cout << "Neural::get_device_type() " << Neural::get_device_type() << endl;

    int num_images = 0, num_labels = 0, img_size = 0, num_rows = 0, num_cols = 0; 
    
    // Load the data
    cout << "Reading proto data" << endl;
    uchar **train_img_data = read_mnist_images("data/train-images-idx3-ubyte", num_images, img_size, num_rows, num_cols);
    uchar *train_labels_data = read_mnist_labels("data/train-labels-idx1-ubyte", num_labels);

    double dml = 1;
    // TODO backprop divide by batch size, emltp? original
    int num_samples = num_images, num_inputs = img_size, num_channels = 1, num_outputs = 10;
    int batch_size = 32, num_hidden_nodes3 = 256;
    
    cout << "Transforming to mono train_images normalized" << endl;
    unique_ptr<double> train_images(data2mono_normalized(train_img_data, num_images, img_size, dml));
    cout << "Transforming to mono train_labels" << endl;
    unique_ptr<double> train_labels(labels2mono1hot(train_labels_data, num_labels, num_outputs));
    cout << "Transforming to mono train_images_orig" << endl;
    unique_ptr<double> train_images_orig(data2mono(train_img_data, num_images, img_size));
    
    // int sst = atoi(argv[1]);
    //TODO layer data allocation control
    //TODO factories?
    //TODO helper structs? e.g size2D, imgData3D?
    
    /////////////////////////////////////////
    cout << "Init test_input" << endl;
    
    int num_s = 5, num_ch = 3, num_r = 3, num_c = 3;
    double *test_input = new double[num_s*num_ch*num_r*num_c];
    
    for(int s = 0; s < num_s; s++) {
        for(int ch=0; ch<num_ch; ch++) {
            for(int i = 0; i < num_r; i++) {
                for(int j = 0; j < num_c; j++) {
                    int val = s*num_ch*num_r*num_c + ch*num_r*num_c + i*num_c + j + 1;
                    int vl = 31*val;
                    int vll = vl%255;
                    test_input[s*num_ch*num_r*num_c + ch*num_r*num_c + i*num_c + j] = (vll -255.0f/2)/255.0f;            
                }
            }

        }
    }
    
    
    vector<vector<cv::Mat>> mdata;
    for(int s = 0; s < num_s; s++) {
        vector<cv::Mat> sdata;
        for(int ch=0; ch<num_ch; ch++) {
            sdata.push_back(cv::Mat(num_r, num_c, CV_64FC1, test_input+s*num_ch*num_r*num_c + ch*num_r*num_c));
        }
        mdata.push_back(sdata);
    }
    for(int i = 0; i < mdata.size(); i++) {
        for(int s = 0; s < mdata[i].size(); s++) {
            cout << mdata[i][s] << endl;
        }
    }
    
    int num_out = 5;
    
    int *test_labels = new int[num_s*num_out];
    
    cout << "Init test labels" << endl;
    for(int s = 0; s < num_s; s++) {
        int lbl = s%num_out;
        
        for(int ot = 0; ot < num_out; ot++) {
            if(lbl == ot) {
                test_labels[s*num_out + ot] = 1;
            }
            else {
                test_labels[s*num_out + ot] = 0;
            }
        }
    }

    cout << "Convert to cv::Mat" << endl;
    cv::Mat mdata_labels(num_s, num_out, CV_32S, test_labels);
    
    cout << "Print labels" << endl;
    cout << mdata_labels << endl;
        
    ////////////////////////////////////////////
    
    cout << "Init ltest_data" << endl;
    Tensor4D<double> * ltest_data = new Tensor4D<double>(test_input, Shape4D(num_s, num_ch, num_r, num_c));
    ltest_data->print();
    
    cout << "Init ltest_labels" << endl;
    Tensor4D<int> * ltest_labels = new Tensor4D<int>(test_labels, Shape4D(num_s, num_out));
    ltest_labels->print();    
    
    int bs = 2;
    
    using Neural::Network;
    using Neural::Layers::Layer;
    cout << "Init network" << endl;
    Network testnet; //destructor?
    
    testnet.add_layer(new Neural::Layers::Conv(Shape4D(bs, num_ch, num_r, num_c), 4, "relu", 2, 1, "same"));
    // testnet.add_layer(new FcLayer(2, "relu"));
    testnet.add_l<Neural::Layers::Fc>(num_out, "softmax");
    
    testnet.set_debug(true);
    testnet.train(*ltest_data, *ltest_labels, bs, 1, true, 0.5, "CrossEntropy");
    
    delete ltest_data;
    delete ltest_labels;
    delete[] test_input;
    delete[] test_labels;
    return 0;
}
