#include <random>
#include <iostream>
#include <fstream>
#include "mnist.hpp"
#include "utils.hpp"

/*
extern void run_optimal(double *, double *, double *, double *, int, int, int, int);
extern void run_w2kernel(double *);
extern void testpr();
*/

using Neural::Tensor4D;
using Neural::LabeledData;
using Neural::Shape4D;
using namespace std;

typedef unsigned char uchar;

template<class T>
Tensor4D<T> * read_mnist_images(string full_path) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        int image_size = n_rows * n_cols;

        Shape4D data_shape(number_of_images, 1, n_rows, n_cols);
        Tensor4D<T> * _dataset = new Tensor4D<T>(data_shape);

        for(int i = 0; i < number_of_images; i++) {
            uchar *__row = new uchar[image_size];
            file.read((char *)__row, image_size);

            for(int r = 0; r < image_size; r++) {
                _dataset->iat(i*image_size + r) = __row[r];
            } 
            delete[] __row;
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

template Tensor4D<double> *read_mnist_images(string full_path);
template Tensor4D<float> *read_mnist_images(string full_path);

Tensor4D<int> * read_mnist_labels(string full_path) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);
    if(file.is_open()) {
        int magic_number = 0, number_of_labels;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        // uchar* _dataset = new uchar[number_of_labels];
        // for(int i = 0; i < number_of_labels; i++) {
        //     file.read((char*)&_dataset[i], 1);
        // }
        Shape4D labels_shape(number_of_labels, 10, 1, 1);
        Tensor4D<int> * _dataset = new Tensor4D<int>(labels_shape);
        
        for(int i = 0; i < number_of_labels; i++) {
            uchar lbl;
            file.read((char*)&lbl, 1);

            int lblint = (int)lbl;
            
            if(i < 32) {
                PLOGD << lblint;
            }
            for(int m = 0; m < 10; m++) {
                int lbl1hot = 0;
                if(m==lblint) {
                    lbl1hot = 1;
                }

                _dataset->iat(i*10 + m) = lbl1hot;
            }
            
        }     
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

/// @brief 
/// @tparam T datatype of dataset (double, float)
/// @param original_data dataset
/// @param original_labels labels in 1-hot encoding
/// @param percentile percentage of dataset to be designated as valid
/// @return 
template<class T>
vector<LabeledData<T>> split_dataset(Tensor4D<T> * original_data , Tensor4D<int> *original_labels, float percentile) {
    LOGI << "split_dataset";  
    Shape4D data_shape = original_data->shape(), labels_shape = original_labels->shape();
    int B = data_shape[0], C = data_shape[1], H = data_shape[2], W = data_shape[3], M = labels_shape[1];
    
    assert(B==labels_shape[0]);

    int CHW = C*H*W;
    int B_train = (1-percentile)*B;
    int B_valid = B - B_train;
    LOGI.printf("B: %d, B_train: %d, B_valid: %d", B, B_train, B_valid);

    Tensor4D<T> *train_data = new Tensor4D<T>(B_train, C, H, W), *valid_data = new Tensor4D<T>(B_valid, C, H, W);
    Tensor4D<int> *train_labels = new Tensor4D<int>(B_train, M, 1, 1), *valid_labels = new Tensor4D<int>(B_valid, M, 1, 1);

    LOGI << "Populating train_data";
    for(int i = 0; i < B_train; i++) {
        for(int j = 0; j < CHW; j++) {
            train_data->iat(i*CHW + j) = original_data->iat(i*CHW + j);
        }
        for(int j = 0; j < M; j++) {
            train_labels->iat(i*M + j) = original_labels->iat(i*M + j);
        }
    }
    
    LOGI << "Populating valid_data";
    for(int i = 0; i < B_valid; i++) {
        for(int j = 0; j < CHW; j++) {
            valid_data->iat(i*CHW + j) = original_data->iat((i + B_train)*CHW + j);
        }
        for(int j = 0; j < M; j++) {
            valid_labels->iat(i*M + j) = original_labels->iat((i + B_train)*M + j);
        }
    }

    LOGV << "/split_data";
    return vector<LabeledData<T>>{LabeledData<T>(train_data, train_labels), LabeledData<T>(valid_data, valid_labels)};

}

template vector<LabeledData<double>> split_dataset<double>(Tensor4D<double> *, Tensor4D<int> *,  float );
template vector<LabeledData<float>> split_dataset<float>(Tensor4D<float> *, Tensor4D<int> *,  float );

