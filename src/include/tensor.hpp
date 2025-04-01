#pragma once
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include "openacc.h"

namespace Neural {
    struct Shape4D {
        std::vector<int> dims{0,0,0,0};

        Shape4D();
        Shape4D(int, int, int, int);
        Shape4D(int, int, int);
        Shape4D(int, int);
        Shape4D(int);
        
        int & operator[](int index) { return dims[index%4]; };
        const int & operator[](int index) const { return dims[index%4]; };
        bool operator==(const Shape4D& other) const;
        
        int size() const;
        int rank() const;
        Shape4D flat(int) const;

        std::string to_string();
    };

    //TODO make template on acc? 2 versions? one serial one accelerated

    template<class T>
    class AccData {
        T *_data;
        int _size;
        int acc_present_counter{0};
        
    public:
        AccData(int size, bool acc) : _size(size) {
            _data = new T[_size];
            
            if(acc) {
                acc_create();
            }
        }
        
        AccData(int size) : AccData(size, false) {}
        
        ~AccData() {
            this->acc_delete();
            delete[] _data;
        }
        
        void acc_create() {
            #pragma acc enter data create(_data[:_size])
            acc_present_counter++;
        }
        
        void acc_copyin() {
            #pragma acc enter data copyin(_data[:_size])
            acc_present_counter++;
        }
        
        void acc_copyout() {
            #pragma acc exit data copyout(_data[:_size])
            acc_present_counter--;
        }
        
        void acc_delete() {
            #pragma acc exit data delete(_data[:_size])
            acc_present_counter--;
        }
        
        
    };

    template<class T>
    class acc_shared_ptr : public std::shared_ptr<T> {
        
        ~acc_shared_ptr() {
            std::shared_ptr::~shared_ptr();
            
            T* tdata = this->get();
            
        }
        //TODO copy ctor,assignment, destructor
    };


    template <class T=double>
    class Tensor4D {
        
    private:
        Shape4D _shape;
        T * _data; //TODO get rid of vector, replace with shared_ptr<double> ?
        bool _allocated{false};
        
        void reset_data(); 
        
    public:
        template<class U>
        Tensor4D(U* cdata, Shape4D cshape) : _shape(cshape) {
            this->reserve();
    
            for(int i = 0; i < size(); i++) {
                _data[i] = (T)cdata[i];
            }
            
        }
        template<class U> Tensor4D(U* cdata, int a, int b, int c, int d) : Tensor4D<U>(cdata, Shape4D(a,b,c,d));
        Tensor4D();

        Tensor4D(Shape4D);
        Tensor4D(int, int, int, int);
        ~Tensor4D(); //destructor
        Tensor4D(const Tensor4D &); //copy ctor
        Tensor4D(Tensor4D &&); //move ctor
            
        //getters
        T* data() { return _data; }
        const T* data() const { return _data; }
        Shape4D shape() const { return _shape; }
        int size() const { return _shape.size(); }
        
        //setters
        void reserve() {
            if(!_allocated) {
                this->_data = new T[this->size()];
                _allocated=true;
            }
        }
        
        std::string to_string();
        Tensor4D<T> & reshape(const Shape4D &new_shape) { assert(new_shape.size() == _shape.size()); _shape = new_shape; return *this; }
        
        std::ostream & put(std::ostream & );
        
        void set(int index, T val) {
            _data[index] = val;
        }
        
        T& at(int i, int j, int k, int l) const {
            return _data[i*_shape[1]*_shape[2]*_shape[3] + j*_shape[2]*_shape[3] + k*_shape[3] + l];
        }
        
        T& iat(int i) const {
            return _data[i];
        }
        
        //acc
        bool is_present_acc() const;
        bool is_present_acc();
        void update_self_acc();

        void create_acc();
        void copyin_acc();
        void copyout_acc();
        void delete_acc();
        
        void print();
        
        //TODO test inline/not-inline routines openacc

        Tensor4D &operator=(const Tensor4D &); //copy assignment
        Tensor4D &operator=(Tensor4D &&); //move assignmnet
    };

    template<class T>
    class LabeledData {
    private: 
        Tensor4D<T> *data;
        Tensor4D<int> *labels;
    public:
        LabeledData(Tensor4D<T> *, Tensor4D<int> *);

        auto get_data() { return data; }
        auto get_labels() { return labels; }
    };
};

void assert_shape(Neural::Shape4D , Neural::Shape4D );
