#pragma once
#include <string>
#include "libraries/tensor.hpp"
#include <random>
#include <vector>
#define DEPSILON 0.5E-15
//#include <cuchar>
/*
extern void run_optimal(double *, double *, double *, double *, int, int, int, int);
extern void run_w2kernel(double *);
extern void testpr();
*/
typedef unsigned char uchar;

template<class T>
Tensor4D<T>* read_mnist_images(std::string full_path) {
    
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) {
            throw std::runtime_error("Invalid MNIST image file!");
        }
    
        int number_of_images, n_rows, n_cols;
        
        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
        
        const int img_size = n_rows*n_cols;
        
        Tensor4D<T>* _dataset = new Tensor4D<T>(Shape4D(number_of_images, 1, n_rows, n_cols));
        
        for(int i = 0; i < number_of_images; i++) {
            uchar *row = new uchar[img_size];
            file.read((char *)row, img_size);
            for(int kl = 0; kl < img_size; kl++) {
                _dataset->at(i, 1, kl) = row[kl];
            }
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

template<class T>
Tensor4D<int>* read_mnist_labels(std::string full_path) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) {
            throw std::runtime_error("Invalid MNIST label file!");
        }
        
        int number_of_labels;
        
        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        Tensor4D<int> *_dataset = new Tensor4D<int>(Shape4D(number_of_labels, 10, 1, 1));
        
        for(int i = 0; i < number_of_labels; i++) {
            int labl;
            file.read((char*)&labl, 1);
            
            for(int l = 0; l < 10; l++) {
                int lbl1hot;
                
                if(l == labl) {
                    lbl1hot = 1;
                }
                else {
                    lbl1hot = 0;
                }
                
                _dataset->at(i, l) = lbl1hot;
            }
            
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

template <class T> T** convert_train_dataset(uchar **, int , int , int );
double *data2mono(uchar **, int, int);
double *data2mono_normalized(uchar **, int , int , double );

template<class T>
T *labels2mono1hot(uchar *labels, int num_images, int num_labels) {
    T *_labels1hot = new T[num_images * num_labels];
    
    for(int i = 0; i < num_images; i++) {
        for(int l = 0; l < num_labels; l++) {
            if((int)labels[i] == l) {
                _labels1hot[i*num_labels + l] = (T)1;
            }
            else {
                _labels1hot[i*num_labels + l] = (T)0;
            }
        }
    }
    
    return _labels1hot;
}

template<class T>
auto split_data(const Tensor4D<T> &data, const Tensor4D<int> labels, float percentile) {
    Shape4D data_shape = data.shape(), labels_shape = labels.shape();
    
    assert(data_shape[0]==labels_shape[0]);
    
    int data_size = data.size(), labels_size = labels.size();
    
    int data_num = data_shape[0];
    int data_sample_size = data_size/data_num, labels_sample_size = labels_size/data_num;
    
    int train_num = (1-percentile)*data_num;
    int valid_num = data_num - train_num;
    
    LabeledData<T> train_data_labeled, valid_data_labeled;
    
    train_data_labeled.data = new Tensor4D<T>(Shape4D(train_num, data_shape[1], data_shape[2], data_shape[3]));
    valid_data_labeled.data = new Tensor4D<T>(Shape4D(valid_num, data_shape[1], data_shape[2], data_shape[3]));
    
    train_data_labeled.labels = new Tensor4D<int>(Shape4D(train_num, labels_shape[1], labels_shape[2], labels_shape[3]));
    valid_data_labeled.labels = new Tensor4D<int>(Shape4D(valid_num, labels_shape[1], labels_shape[2], labels_shape[3]));
    
    for(int i = 0; i < train_num; i++) {
        for(int j = 0; j < data_sample_size; j++) {
            train_data_labeled.data->set(i*data_sample_size + j, data.at(i*data_sample_size + j));
        }
        for(int j = 0; j < labels_sample_size; j++) {
            train_data_labeled.labels->set(i*labels_sample_size + j, labels.at(i*labels_sample_size + j));
        }
    }
    
    for(int i = 0; i < train_num + valid_num; i++) {
        for(int j = 0; j < data_sample_size; j++) {
            valid_data_labeled.data->set(i*data_sample_size + j, data.at(i*data_sample_size + j));
        }
        for(int j = 0; j < labels_sample_size; j++) {
            valid_data_labeled.labels->set(i*labels_sample_size + j, labels.at(i*labels_sample_size + j));
        }
    }
    
    return std::vector<LabeledData<T>>{train_data_labeled, valid_data_labeled};
}
