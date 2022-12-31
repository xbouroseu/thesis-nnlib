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

template<class T>
vector<LabeledData<T>> split_dataset(Tensor4D<T> * original_data , Tensor4D<int> *original_labels, float percentile) {
    LOG("split_dataset");  
    Shape4D data_shape = original_data->shape(), labels_shape = original_labels->shape();

    assert(data_shape[0]==labels_shape[0]);
        
    int data_sample_size = data_shape[1]*data_shape[2]*data_shape[3], labels_sample_size = labels_shape[1]*labels_shape[2]*labels_shape[3];
    
    int num_train_data = (1-percentile)*data_shape[0];
    int num_valid_data = data_shape[0] - num_train_data;
    
    Shape4D train_data_shape(data_shape), valid_data_shape(data_shape);
    Shape4D train_labels_shape(labels_shape), valid_labels_shape(labels_shape);
    train_data_shape[0] = num_train_data;
    train_labels_shape[0] = num_train_data;
    valid_data_shape[0] = num_valid_data;
    valid_labels_shape[0] = num_valid_data;

    LabeledData<T> train_data_labeled, valid_data_labeled;
    
    train_data_labeled.data = new Tensor4D<T>(train_data_shape);
    train_data_labeled.labels = new Tensor4D<int>(train_labels_shape);

    valid_data_labeled.data = new Tensor4D<T>(valid_data_shape);
    valid_data_labeled.labels = new Tensor4D<int>(valid_labels_shape);
    
    for(int i = 0; i < num_train_data; i++) {
        for(int j = 0; j < data_sample_size; j++) {
            train_data_labeled.data->iat(i*data_sample_size + j) = original_data->iat(i*data_sample_size + j);
        }
        for(int j = 0; j < labels_sample_size; j++) {
            train_data_labeled.labels->iat(i*labels_sample_size + j) = original_labels->iat(i*labels_sample_size + j);
        }
    }
    
    for(int i = num_train_data; i < num_train_data + num_valid_data; i++) {
        for(int j = 0; j < data_sample_size; j++) {
            valid_data_labeled.data->iat(i*data_sample_size + j) = original_data->iat(i*data_sample_size + j);
        }
        for(int j = 0; j < labels_sample_size; j++) {
            valid_data_labeled.labels->iat(i*labels_sample_size + j) = original_labels->iat(i*labels_sample_size + j);
        }
    }

    LOG("/split_data");
    return vector<LabeledData<T>>{train_data_labeled, valid_data_labeled};

}

template vector<LabeledData<double>> split_dataset<double>(Tensor4D<double> *, Tensor4D<int> *,  float );
template vector<LabeledData<float>> split_dataset<float>(Tensor4D<float> *, Tensor4D<int> *,  float );

template <class T>
T** convert_train_dataset(uchar **dataset, int num_images, int num_rows, int num_cols) {
    T** _cdata = new T*[num_images];
    
    for(int k = 0; k < num_images; k++) {
        _cdata[k] = new T[num_rows * num_cols];
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                _cdata[k][28*i + j] = (T)(dataset[k][28*i+j]);
            }
        }
    }
    
    return _cdata;
}

double *data2mono(uchar **data, int num_images, int img_size) {
    double *_data = new double[num_images * img_size];
    
    for(int i = 0; i < num_images; i++) {
        for(int k = 0; k < img_size; k++) {
            _data[i * img_size + k] = (double)(data[i][k]);
        }
    }
    
    return _data;
}

double *data2mono_normalized(uchar **data, int num_images, int img_size, double dml) {
    double *_data = new double[num_images * img_size];
    
    for(int i = 0; i < num_images; i++) {
        for(int k = 0; k < img_size; k++) {
            _data[i * img_size + k] = dml * ((double)(data[i][k] - 255.0f/2))/255.0f;
        }
    }
    
    return _data;
}

template<class T>
T *labels2mono1hot(unsigned char *labels, int num_images, int num_labels) {
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

template int* labels2mono1hot<int>(unsigned char *, int, int);
template double* labels2mono1hot<double>(unsigned char *, int, int);
template float* labels2mono1hot<float>(unsigned char *, int, int);

// template<class T>
// vector<Tensor4D<T> *> split_random(const Tensor4D<T> &data, const Tensor4D<T> labels) {
//     Shape4D data_shape = data.shape(), labels_shape = labels.shape();
    
//     assert(data_shape[0]==labels_shape[0]);
    
//     random_device rd; // obtain a random number from hardware
//     mt19937 gen(rd()); // seed the generator
//     uniform_int_distribution<> distr(0, data_shape[0]); // define the range
// }

// template vector<Tensor4D<int> *> split_random<int>(const Tensor4D<int> &, const Tensor4D<int>);
// template vector<Tensor4D<double> *> split_random<double>(const Tensor4D<double> &, const Tensor4D<double>);
// template vector<Tensor4D<float> *> split_random<float>(const Tensor4D<float> &, const Tensor4D<float>);



