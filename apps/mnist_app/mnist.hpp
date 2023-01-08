#pragma once
#include <string>

#include <vector>
#include "tensor.hpp"

#define DEPSILON 0.5E-15

template <class T> Neural::Tensor4D<T> *read_mnist_images(std::string);
Neural::Tensor4D<int> *read_mnist_labels(std::string);
template<class T> std::vector<Neural::LabeledData<T>> split_dataset(Neural::Tensor4D<T> *  , Neural::Tensor4D<int> *, float );
