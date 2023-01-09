#include <string>
#include <unistd.h>
#include <iostream>
#include "openacc.h"

using namespace std;

template <typename T> 
constexpr auto type_name() {
    std::string_view name, prefix, suffix;
    
    #ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
    #elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
    #elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
    #endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    
    return name;
}

int main() {
    printf("Current working dir: %s\n", get_current_dir_name());
    cout << "__FILE__" << __FILE__ << endl;
    cout << "acc_get_device_type() " << acc_get_device_type() << endl;

    int *a = new int[10];
    a[2] = 3;

    // cout << "Creating data" << endl;
    // #pragma acc enter data create(a[:10])

    cout << "Deleting data" << endl;
    #pragma acc exit data delete(a[:10])
}
