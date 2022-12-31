#pragma once
#include <string>

#include <vector>
#include "tensor.hpp"

#define DEPSILON 0.5E-15
//#include <cunsigned char>
/*
extern void run_optimal(double *, double *, double *, double *, int, int, int, int);
extern void run_w2kernel(double *);
extern void testpr();
*/


template <class T> Neural::Tensor4D<T> *read_mnist_images(std::string);
Neural::Tensor4D<int> *read_mnist_labels(std::string);
template<class T> std::vector<Neural::LabeledData<T>> split_dataset(Neural::Tensor4D<T> *  , Neural::Tensor4D<int> *, float );

template <class T> T** convert_train_dataset(unsigned char **, int , int , int );
double *data2mono(unsigned char **, int, int);
double *data2mono_normalized(unsigned char **, int , int , double );
template <class T>
T *labels2mono1hot(unsigned char *, int, int);

