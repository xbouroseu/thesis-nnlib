#pragma once
#ifdef _OPENACC
constexpr int IS_OPENACC = 1;
#include "openacc.h"
#define mdevicetype() (int)acc_get_device_type()
#define mdeviceptr(mptr) acc_deviceptr(mptr)
#define misacc() true
#define mispresent(_mdata, _n) acc_is_present(_mdata, _n)
#else
constexpr int IS_OPENACC = 0;
#define mdevicetype() 2
#define mdeviceptr(mptr) mptr
#define misacc() false
#define mispresent(_mdata, _n) true
#endif
#define hdevicetype 2
#include <cstddef>

#ifdef _DEBUG
#include <iostream>
#define LOG(m)  std::cout << m << std::endl;
#else
#define LOG(m)  
#endif

#include <ctime>
double dur(double );

namespace Neural {
    constexpr int device_type_host = 2;
    constexpr int device_type_gpu = 4;
    
    constexpr bool is_acc() {
        return misacc();
    }
    
    int get_device_type();
    void *deviceptr(void*);
    
    bool is_present(void *, std::size_t);
    bool is_present(const void *, std::size_t);
    
    //forward declarations
    class Network;
    
    namespace Layers {
        class Layer;
        class Weighted;
        class Fc;
        class Conv;
    }
};
