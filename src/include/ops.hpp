#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "tensor.hpp"
#include "utils.hpp"

#if !defined(_OPENACC)
#define SAFEDATA
#endif

constexpr int DEBUG__ = 1;

void tparallel_conv5(double *conv_input, double *conv_filters, double *conv_output, int batch_size, int in_channels, int in_height, int in_width, int out_channels , int out_height, int out_width, int filter_size, int stride, bool debug);

template<class T> void acc_copy(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_add(Neural::Tensor4D<T> *, const Neural::Tensor4D<T> &);
template<class T> void acc_val(Neural::Tensor4D<T> *, T );
template<class T> void acc_zeros(Neural::Tensor4D<T> *);
template<class T> void acc_mltp(Neural::Tensor4D<T> *, T );
template<class T> void acc_accumulate(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_rng(Neural::Tensor4D<T> *, T );
template<class T> void acc_flip_spatial(Neural::Tensor4D<T> *);
template<class T> void acc_matrix_multiply(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_convolution2D(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *, const std::vector<int> &);
template<class T> void acc_relu(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_relu_backprop(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_sigmoid(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_sigmoid_backprop(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_softmax(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_softmax_backprop(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
template<class T> void acc_pad2D_inner(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *, int , int , int , int , int , int );
template<class T> void acc_pad2D(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *, int , int , int , int );
template<class T> Neural::Tensor4D<T>* acc_padded2D_inner(const Neural::Tensor4D<T> &, int , int , int , int , int , int );
template<class T> void acc_rev_pad2D(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *, int , int , int , int );
template<class T> void acc_normalize_img(Neural::Tensor4D<T> *);
template <class T> void acc_make_batch(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *, int );
template<class T> Neural::Tensor4D<int> * acc_calc_confusion_matrix(Neural::Tensor4D<T> &, Neural::Tensor4D<int> &);

//comment 4
namespace Neural {
    namespace Activations {
        
        template<class T>
        class Base {
            std::string _name;
            void (*_fn)(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
            void (*_backfn)(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *);
            
            public:
                Base() {}
                Base(std::string name, void (*fn)(const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *), void (*backfn)(const Neural::Tensor4D<T> &, const Neural::Tensor4D<T> &, Neural::Tensor4D<T> *)  ) : _name(name), _fn(fn), _backfn(backfn) {}
                
                std::string name() { return _name; }
                
                void apply(const Neural::Tensor4D<T> &input, Neural::Tensor4D<T> *output) {
                    _fn(input, output);
                }
                
                void backward(const Neural::Tensor4D<T> &drv_error_output, const Neural::Tensor4D<T> &output, Neural::Tensor4D<T> *drv_error_output_preact) {
                    _backfn(drv_error_output, output, drv_error_output_preact);
                }
        };
        
        const Base<double> Relu("relu", acc_relu<double>, acc_relu_backprop<double>);
        const Base<double> Softmax("softmax", acc_softmax<double>, acc_softmax_backprop<double>);
        const Base<double> Sigmoid("sigmoid", acc_sigmoid<double>, acc_sigmoid_backprop<double>);
    }
}


template<class T, int DIM>
void AddVecDim(Neural::Tensor4D<T> *A, const Neural::Tensor4D<T> &B) {
    Neural::Shape4D a_shape = A->shape(), b_shape = B.shape();
    
    LOGD << "a_shape: " << a_shape.to_string();
    LOGD << "b_shape: " << b_shape.to_string();
    
    if constexpr(DIM==0) {
        if(!(b_shape[1]==1 && b_shape[2]==1 && b_shape[3]==1)) {
            std::string emsg = "Error: Tensor B is not Vector(ax1x1x1)";
            throw(std::invalid_argument(emsg));
        }
    }
    else if constexpr(DIM==1) {
        if(!(b_shape[0]==1 && b_shape[2]==1 && b_shape[3]==1)) {
            std::string emsg = "Error: Tensor B is not Vector(1xbx1x1)";
            throw(std::invalid_argument(emsg));
        }
    }
    else if constexpr(DIM==2) {
        if(!(b_shape[0]==1 && b_shape[1]==1 && b_shape[3]==1)) {
            std::string emsg = "Error: Tensor B is not Vector(1x1xcx1)";
            throw(std::invalid_argument(emsg));
        }
    }
    else if constexpr(DIM==3) {
        if(!(b_shape[0]==1 && b_shape[1]==1 && b_shape[2]==1)) {
            std::string emsg = "Error: Tensor B is not Vector(1x1x1xd)";
            throw(std::invalid_argument(emsg));
        }
    }
    
    if(b_shape[DIM] != a_shape[DIM]) {
        std::string emsg = "Error: Tensor a_shape[" + std::to_string(DIM) + "] != b_shape[" + std::to_string(DIM) + "]";
        throw(std::invalid_argument(emsg));
    }
    
    int sizeA = a_shape.size(), sizeB = b_shape.size(), a=a_shape[0], b=a_shape[1], c=a_shape[2], d=a_shape[3];
    int bcd = b*c*d, cd = c*d;
    T *a_data = A->data();
    const T *b_data = B.data();
    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop collapse(4)
    for(int i=0; i<a; i++) {
        for(int j=0; j<b; j++) {
            for(int k=0; k<c; k++) {
                for(int l=0; l<d ; l++) {
                    if constexpr(DIM == 0) {
                        a_data[i*bcd + j*cd + k*d + l] += b_data[i];
                    }
                    else if constexpr(DIM == 1) {
                        a_data[i*bcd + j*cd + k*d + l] += b_data[j];
                    }
                    else if constexpr(DIM == 2) {
                        a_data[i*bcd + j*cd + k*d + l] += b_data[k];
                    }
                    else if constexpr(DIM == 3) {
                        a_data[i*bcd + j*cd + k*d + l] += b_data[l];
                    }
                }
            }
        }
    }
    }
}

template<class T, int DIM1, int DIM2>
Neural::Tensor4D<T>* acc_transposed(const Neural::Tensor4D<T> &input) {
    Neural::Shape4D a_shape = input.shape(), t_shape = Neural::Shape4D(a_shape);
    
    static_assert(DIM2>DIM1);
    static_assert(DIM1>=0 && DIM1 < 4);
    static_assert(DIM2>=0 && DIM2 < 4);
    
    int a_mlt[4]{a_shape[1]*a_shape[2]*a_shape[3], a_shape[2]*a_shape[3], a_shape[3], 1};
    int at_mlt[4]{a_mlt[0], a_mlt[1], a_mlt[2], a_mlt[3]};
    
    t_shape[DIM2] = a_shape[DIM1];
    t_shape[DIM1] = a_shape[DIM2];
    
    at_mlt[DIM2] = a_mlt[DIM1];
    at_mlt[DIM1] = a_mlt[DIM2];
    
    Neural::Tensor4D<T> *ret = new Neural::Tensor4D<T>(t_shape);
    ret->create_acc();
    
    const T* a_data = input.data();
    T *t_data = ret->data();
    
    int AA = a_shape[0], AB = a_shape[1], AC = a_shape[2], AD = a_shape[3];
    int TA = t_shape[0], TB = t_shape[1], TC = t_shape[2], TD = t_shape[3];
    
    #pragma acc parallel loop collapse(4) present(a_data[:AA*AB*AC*AD], t_data[:TA*TB*TC*TD])
    for(int a = 0; a < TA; a++) {
        for(int b = 0; b < TB; b++) {
            for(int c = 0; c < TC; c++) {
                for(int d = 0; d < TD; d++) {                    
                    t_data[a*TB*TC*TD + b*TC*TD + c*TD + d] = a_data[a*at_mlt[0] + b*at_mlt[1] + c*at_mlt[2] + d*at_mlt[3]];                 
                }
            }
        }
    }
    
    return ret;
}
