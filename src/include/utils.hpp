#pragma once
#include <plog/Log.h>
#include <ctime>

#define _LLOG_A(_lvl, _what, _prens) IF_PLOG(plog::_lvl) { std::cout << "[" << _prens << "]\n" << _what->to_string() << std::endl; }
#define _LLOG(_lvl, _what) _LLOG_A(_lvl, _what, #_what)

#ifdef _OPENACC
constexpr int IS_OPENACC = 1;
#include "openacc.h"
#define mdevicetype() (int)acc_get_device_type()
#define mdeviceptr(mptr) acc_deviceptr(mptr)
#define misacc true
#define mispresent(_mdata, _n) acc_is_present(_mdata, _n)
#else
constexpr int IS_OPENACC = 0;
#define mdevicetype() 2
#define mdeviceptr(mptr) mptr
#define misacc false
#define mispresent(_mdata, _n) true
#endif
#define hdevicetype 2

namespace plog
{
    class MyFormatter
    {
    public:
        static util::nstring header() { return util::nstring(); };
        static util::nstring format(const Record& record) {
            util::nostringstream ret;
            ret << PLOG_NSTR("[") << record.getFunc() << PLOG_NSTR(":") << record.getLine() << PLOG_NSTR("] ") << record.getMessage() << PLOG_NSTR("\n");
            return ret.str();
        };
    };
}

template <class... Args>
double timeop(void (*fcnptr)(), Args... args) {
    clock_t start = clock();
    (*fcnptr)(args);
    double duration = (clock()-start)/CLOCKS_PER_SEC;
    return duration;
}

double dur(clock_t);

namespace Neural {
    constexpr int device_type_host = 2;
    constexpr int device_type_gpu = 4;
    
    constexpr bool is_acc() {
        return misacc;
    }

    int get_device_type();
    void *deviceptr(void*);
    
    bool is_present(void *, std::size_t);
    bool is_present(const void *, std::size_t);
};
