#include "utils.hpp"
#include <cstddef>
#include <type_traits>
#include <ctime>
#include <iostream>

using namespace std;

double dur(clock_t start) {
    double diff = clock()-start;
    return diff/CLOCKS_PER_SEC;
}

int Neural::get_device_type() {
    return mdevicetype();
}

void* Neural::deviceptr(void* hptr) {
    return mdeviceptr(hptr);
}

bool Neural::is_present(void *ptr, size_t _size) {
    return mispresent(ptr, _size);
}

bool Neural::is_present(const void *ptr, size_t _size) {
    return mispresent((void*) ptr, _size);
}

