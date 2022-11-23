#include "tensor.h"
#include "neural.h"
#include <vector>
#include <iostream>

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

int Shape4D::rank() const {
    for(int i = 3; i >= 0; i--) {
        if(dims[i] != 1) {
            return i+1;
        }
    }
    
    return 0;
}

void Shape4D::reshape(Shape4D _shape) {
    for(int i = 0; i < 4; i++) {
        dims[i] = _shape.dims[i];
    }
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

void Shape4D::flatten(int dim) {
    reshape(flat(dim));
}

template<class T> Tensor4D<T>::Tensor4D() {}

template<class T> Tensor4D<T>::Tensor4D(const T* _data, Shape4D _shape) : shape(_shape) {
    data = new T[shape.size()];
    for(int i = 0; i < shape.size(); i++) {
        data[i] = _data[i];
    }
}

template<class T> Tensor4D<T>::Tensor4D(const T* _data, int a, int b, int c, int d) : Tensor4D(_data, Shape4D(a,b,c,d)) {}

template<class T> Tensor4D<T>::Tensor4D(Shape4D _shape) : shape(_shape) {
    data = new T[shape.size()];
}

template<class T> Tensor4D<T>::Tensor4D(int a, int b, int c, int d) : Tensor4D(Shape4D(a,b,c,d)) {}


template<class T> Tensor4D<T>::~Tensor4D() {    
    deaccel();

    delete[] data;
}

//TODO Implicit inline?
template<class T>
inline T& Tensor4D<T>::operator[](int index) const {
    return data[index];
}


//TODO copyin?

template<class T> Tensor4D<T>* Tensor4D<T>::accel() {
    #pragma acc enter data pcopyin(this) pcopyin(data[:shape.size()])
    return this;
}

//TODO copyout?
template<class T> Tensor4D<T>* Tensor4D<T>::deaccel() {
    #pragma acc exit data delete(data[:shape.size()]) delete(this) finalize
    return this;
}

template<class T> void Tensor4D<T>::acc_update_self() {
    #pragma acc update self(this->data[:this->shape.size()]) if(is_present_gpu())
}

template<class T> inline void Tensor4D<T>::set_inline(int index, T val) {
    data[index] = val;
}

template<class T> bool Tensor4D<T>::is_present_gpu() {
    return neural_is_present_gpu(data, shape.size());
}

#pragma acc routine seq
double TensorGet(Tensor4D<double> &Tens, int ind){
    return Tens.data[ind];
}

template<class T> void Tensor4D<T>::print() {
    if(acc) {
        this->acc_update_self();
    }
    
    int a = shape[0], b = shape[1], c = shape[2], d = shape[3];
    printf("Size: %d x %d x %d x %d = %d\n", a, b, c, d, shape.size());
    
    for(int nn = 0; nn < a; nn++) {
        printf("\n");
        for(int dd = 0; dd < b; dd++) {
            printf("----------\n");
            for(int nr = 0; nr < c; nr++) {
                for(int nc = 0; nc < d; nc++) {
                    printf("%+4.5f|", data[nn* b * c * d + dd *c*d +  nr * d + nc]);
                }
                printf("\n");
            }
            printf("---------");
        }
    }
    printf("\n");
}

template class Tensor4D<double>;
template class Tensor4D<float>;
