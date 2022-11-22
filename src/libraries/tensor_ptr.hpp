#pragma once
#include <vector>
#include <memory>

struct Shape4D {
    std::vector<int> dims{0,0,0,0};

    Shape4D();
    Shape4D(int, int, int, int);
    Shape4D(int, int, int);
    Shape4D(int, int);
    Shape4D(int);
    
    int operator[](int index) const { return dims[index%4]; };
    
    bool operator==(const Shape4D& other) const {
        bool eq = true;
        for(int i = 0; i < 4; i++) {
            if(dims[i] != other[i]) {
                eq = false;
                break;
            }
        }
        
        return eq;
    }
    int size() const { return dims[0]*dims[1]*dims[2]*dims[3]; };
    int rank() const;
    Shape4D flat(int) const;
    void flatten(int);
    
    void reshape(Shape4D);
    void reshape(int, int, int, int);    
};

template <class T=double>
class Tensor4D {
    
friend double TensorGet(Tensor4D<double> &, int);

private:
    Shape4D shape;
    T* data;
    bool acc{false}, copyin{false}, debug{false};

public:
    Tensor4D();
    Tensor4D(const T*, Shape4D);
    Tensor4D(const T*, int, int, int, int);
    Tensor4D(Shape4D);
    Tensor4D(int, int, int, int);
    ~Tensor4D();
    
    bool is_present_gpu();
    T* get_data() const { return data; }
    Shape4D get_shape() const { return shape; }
    
    int size() const { return shape.size(); }
    
    void flatten(int dim) { shape.flatten(dim); }
    
    #pragma acc routine seq
    T& operator[](int) const;
    
    #pragma acc routine seq
    void set_inline(int, T);
    
    #pragma acc routine seq
    void set(int index, T val) {
        data[index] = val;
    }
    
    #pragma acc routine seq
    void set_atomic(int index, T val) {
        #pragma acc atomic write
        data[index] = val;
    }
    
    #pragma acc routine seq
    T get(int index) const {
        return data[index];
    }
    
    #pragma acc routine seq
    void add(int index, T val) {
        data[index] += val;
    }

    #pragma acc routine seq
    void mltp(int index, T val) {
        data[index] *= val;
    }
    
    Tensor4D<T>* alloc();
    Tensor4D<T>* dealloc();
    Tensor4D<T>* accel();
    Tensor4D<T>* deaccel();
    void acc_update_self();

    void print();
};

#pragma acc routine seq
double TensorGet(Tensor4D<double> &Tens, int ind);

