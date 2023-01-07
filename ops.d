ops.o ops.d : src/ops.cpp \
  /opt/nvidia/hpc_sdk/Linux_x86_64/22.3/math_libs/include/curand.h \
  src/include/utils.hpp src/include/ops.hpp src/include/tensor.hpp 
