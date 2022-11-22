#include "mnist.hpp"

/*
extern void run_optimal(double *, double *, double *, double *, int, int, int, int);
extern void run_w2kernel(double *);
extern void testpr();
*/
using namespace std;

typedef unsigned char uchar;


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
vector<Tensor4D<T> *> split_random(const Tensor4D<T> &data, const Tensor4D<T> labels) {
    Shape4D data_shape = data.shape(), labels_shape = labels.shape();
    
    assert(data_shape[0]==labels_shape[0]);
    
    random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_int_distribution<> distr(0, data_shape[0]); // define the range
}


