INCLUDE_DIR = src/include
SRC_DIR = src
LIB_DIR = lib

CXX = nvc++
CXXFLAGS = --c++17 -I$(INCLUDE_DIR) -Mcudalib=curand

# SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TARGETS := $(notdir $(basename $(SRCS)))
DEPS := $(addsuffix .d, $(TARGETS))
OBJS := $(addsuffix .o, $(addprefix $(LIB_DIR)/, $(TARGETS)))

# LDFLAGS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
LDFLAGS = -Wl, -Mcudalib=curand

#################  ####################
APPS = mnist sample
ACCLEVELS = noacc acchost acc

# all: lib
# 	cd mnist_app && $(MAKE) all
# 	cd sample_app && $(MAKE) all

# lib: $(foreach acclvl, $(ACCLEVELS), lib_$(acclvl))

test: lvl/one/two/test

# $(CXX) -c $< $(CXXFLAGS) -MMD -o lib/ops.o
lvl/one/two/test: src/ops.cpp
	@echo $(SRCS)
	@echo $(TARGETS)
	@echo $(DEPS)
	@echo $(OBJS)
	@echo "hello"

$(DEPS):%.d:$(SRC_DIR)/%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -c $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

include $(DEPS)
# .PHONY : clean
# clean:
# 	rm lib/*

# clean_all: clean
# 	cd mnist_app && $(MAKE) clean
# 	cd sample_app && $(MAKE) clean
################# / ####################

