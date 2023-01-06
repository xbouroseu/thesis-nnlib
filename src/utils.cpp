#include "utils.hpp"
#include <cstddef>
#include <type_traits>
#include <iostream>

using namespace std;

double dur(double start) {
    return (clock()-start)/CLOCKS_PER_SEC;
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

