#include "tensor.hpp"
#include "neural.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include "openacc.h"

using namespace std;

Shape4D::Shape4D() {}

Shape4D::Shape4D(int a, int b, int c, int d) {
    dims[0] = a;
    dims[1] = b;
    dims[2] = c;
    dims[3] = d;
}

Shape4D::Shape4D(int a, int b, int c) : Shape4D(a, b, c, 1) {}
Shape4D::Shape4D(int a, int b) : Shape4D(a, b, 1) {}
Shape4D::Shape4D(int a) : Shape4D(a, 1) {}

bool Shape4D::operator==(const Shape4D& other) const {
    bool eq = true;
    for(int i = 0; i < 4; i++) {
        if(dims[i] != other.dims[i]) {
            eq = false;
            break;
        }
    }
    
    return eq;
}
    
int Shape4D::size() const {
    return dims[0]*dims[1]*dims[2]*dims[3];
}
int Shape4D::rank() const {
    for(int i = 3; i >= 0; i--) {
        if(dims[i] != 1) {
            return i+1;
        }
    }
    
    return 0;
}

string Shape4D::to_string() {
    return "Shape4D(" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "," + std::to_string(dims[3]) + ")";
}

Shape4D Shape4D::flat(int dim) const {
    int _size = size();
    Shape4D retsh;
    
    if(dim == 0) {
        retsh = Shape4D(_size);
    }
    else if(dim == 1) {
        retsh = Shape4D(dims[0], dims[1]*dims[2]*dims[3]);
    }
    else if(dim == 2) {
        retsh = Shape4D(dims[0], dims[1], dims[2]*dims[3]);
    }
    else if(dim == 3) {
        retsh = *this;
    }
    else {
        throw(std::invalid_argument("Error: not valid flat dim argument"));
    }
    
    return retsh;
}

//TODO more constructors with vector? etc..
template<class T> Tensor4D<T>::Tensor4D(T* cdata, Shape4D cshape) : _shape(cshape) {
    this->reserve();
    
    for(int i = 0; i < size(); i++) {
        _data[i] = cdata[i];
    }
    
    //TODO neural is present? on _data? too? deep copy on gpu?
}

template<class T> Tensor4D<T>::Tensor4D(T* cdata, int a, int b, int c, int d) : Tensor4D(cdata, Shape4D(a,b,c,d)) {}

template<class T> Tensor4D<T>::Tensor4D(Shape4D cshape) :_shape(cshape) {
    LOG("<Tensor4D(" << cshape.to_string() << ", " << __alloc__ << ", " << __acc__ << ")>");
    
    this->reserve();
        
    LOG("</Tensor4D>");
}

template<class T> Tensor4D<T>::Tensor4D(int a, int b, int c, int d) : Tensor4D(Shape4D(a,b,c,d)) {}

template<class T> Tensor4D<T>::Tensor4D() {}

template<class T> Tensor4D<T>::~Tensor4D() {
    LOG("<~Tensor4D>shape: " << _shape.to_string());
    reset_data();
    LOG("</~Tensor4D>");
}

//TODO can delegate other ctor (T*, Shape4D) if same functionality?\
//copy ctor
template<class T> Tensor4D<T>::Tensor4D(const Tensor4D &other) : _shape(other._shape), _allocated(other._allocated) {
    reset_data();
    _data = new T[this->size()];
    
    const T *odata = other._data;
    int osize = this->size();
    bool other_is_present = other.is_present_acc();
    
    if(other_is_present) {
        this->create_acc();
    }
    
    #pragma acc parallel loop present(_data[:osize], adata[:osize]) if(other_is_present)
    for(int i = 0; i < osize; i++) {
        _data[i] = odata[i];
    }
}

//move ctor
//TODO if not & does use count increase?
template<class T> Tensor4D<T>::Tensor4D(Tensor4D &&other) : _shape(other._shape), _allocated(other._allocated) {
    reset_data();
    
    this->_data = other.data();
    
    other._data = nullptr;
}

//copy assignment
template<class T> Tensor4D<T> & Tensor4D<T>::operator=(const Tensor4D &other) {
    reset_data();
    
    this->_shape = other._shape;
    this->_data = new T[this->size()];
    this->_allocated = other._allocated;
    
    const T *odata = other._data;
    int osize = this->size();
    bool other_is_present = other.is_present_acc();
    
    if(other_is_present) {
        this->create_acc();
    }
    
    #pragma acc parallel loop present(_data[:osize], adata[:osize]) if(other_is_present)
    for(int i = 0; i < osize; i++) {
        _data[i] = odata[i];
    }
    
    return *this;
}

//move assignmnet
template<class T> Tensor4D<T> & Tensor4D<T>::operator=(Tensor4D &&other) {
    reset_data();
    
    this->_shape = other.shape();
    this->_data = other._data;
    this->_allocated = other._allocated;
    
    other._data = nullptr;
    return *this;
}

template<class T> bool Tensor4D<T>::is_present_acc() const {
    return Neural::is_present(this->data(), this->size() * sizeof(T));
}

template<class T> void Tensor4D<T>::update_self_acc() {
    LOG("Tensor4D::update_self_acc()");
    LOG("Update self is_present_gpu: ");
    
    bool is_pr = this->is_present_acc();
    LOG(is_pr);
    int _size = this->size();
    #pragma acc update self(_data[:_size]) if(is_pr)
}

template<class G>
void print_line(int __C, int __W) {
    int __z;
    
    if constexpr(is_same<G, int>::value) {
        __z = 6;
    }
    else {
        __z = 12;
    }
    
    for(int c = 0; c < __C; c++) {
        printf("-");
        for(int w = 0; w < __W; w++) {
            for(int z = 0; z < __z; z++) {
                printf("-");
            }
            // printf("------------");
        }
        printf("  ");
    }
}

template<class T> void Tensor4D<T>::print() {
#ifdef _DEBUG
    LOG("Tensor4D::print()");
    this->update_self_acc();
    
    LOG("Shape: " << shape().to_string());
    
    int B = _shape[0], C = _shape[1], H = _shape[2], W = _shape[3];
    
    printf("Size: %d x %d x %d x %d = %d\n", B, C, H, W, _shape.size());
    
    for(int b = 0; b < B; b++) {
        printf("\n");
        print_line<T>(C, W);
        printf("\n");
        for(int h = 0; h < H; h++) {
            for(int c = 0; c < C; c++) {
                printf("|");
                for(int w = 0; w < W; w++) {
                    if constexpr(is_same<T, int>::value) {
                        printf("%5d|", _data[ ( (b* C + c)* H +  h) * W + w]);
                    }
                    else {
                        printf("%+011.5f|", _data[ ( (b* C + c)* H +  h) * W + w]);
                    }
                }
                printf("  ");
            }
            printf("\n");
        }
        print_line<T>(C, W);
        printf("\n");
    }
    printf("\n");
#endif
}

template class Tensor4D<double>;
template class Tensor4D<float>;
template class Tensor4D<int>;
