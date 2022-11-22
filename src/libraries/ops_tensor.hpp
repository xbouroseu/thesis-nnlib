#pragma once
#include "tensor.h"
#include <vector>

void acc_update_self(double *, int);

int vecProduct(const std::vector<int> &);

void make_batch(double *, double *, int , int , int );
void calc_conv_sizes(int, int , int , int , bool , int &, int &, int &, int &, int &, int &);

double *makeweights(int, double);
double *makebiases(int, double);

void transpose(double *, double *, int, int);

void emltp(double *, double , int );

template<class T> void acc_rng(Tensor4D<T> & , T);
template<class T> void acc_val(Tensor4D<T> &, T);
template<class T> void acc_zeros(Tensor4D<T> &);
template<class T> void acc_mltp(Tensor4D<T> &, T);

template<class T> void acc_relu(const Tensor4D<T> &, Tensor4D<T> &, Tensor4D<T> &);
template<class T> void acc_sigmoid(const Tensor4D<T> &, Tensor4D<T> &, Tensor4D<T> &);
template<class T> void acc_softmax(const Tensor4D<T> &, Tensor4D<T> &);

template<class T> void acc_matrix_multiply(const Tensor4D<T> &, const Tensor4D<T> &, Tensor4D<T> &);
template<class T> void acc_convolution2D(const Tensor4D<T> &, const Tensor4D<T> &, Tensor4D<T> &, std::vector<int>);

template<class T> void acc_pad2D(const Tensor4D<T> &, Tensor4D<T> &, int , int , int , int );
template<class T> void acc_make_batch(Tensor4D<T> &, Tensor4D<T> &, int);

template<class T> void AddVecDim0(Tensor4D<T> &, const Tensor4D<T> &);
template<class T> void AddVecDim1(Tensor4D<T> &, const Tensor4D<T> &);
template<class T> void AddVecDim2(Tensor4D<T> &, const Tensor4D<T> &);
template<class T> void AddVecDim3(Tensor4D<T> &, const Tensor4D<T> &);
template<class T> void AddVecDim(Tensor4D<T> &, const Tensor4D<T> &, int);
template<class T> void AddVecBroadCast(Tensor4D<T> &, const Tensor4D<T> &, int);
