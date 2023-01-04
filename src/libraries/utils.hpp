#pragma once
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <ctime>

#define LOG(m) BOOST_LOG_TRIVIAL(m)
#define XLOG(m) if(m > Neural::log_level) { cout }
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
//hello
double dur(double);

namespace Neural {
    int log_level = 0;
    constexpr int device_type_host = 2;
    constexpr int device_type_gpu = 4;
    
    constexpr bool is_acc() {
        return misacc();
    }
    
    void set_log_level(boost::log::trivial::severity_level);
    int get_device_type();
    void *deviceptr(void*);
    
    bool is_present(void *, std::size_t);
    bool is_present(const void *, std::size_t);
};
