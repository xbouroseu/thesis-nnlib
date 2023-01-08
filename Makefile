SRC_DIR = src
INCLUDE_DIR = src/include
LIB_DIR = lib

CXX = nvc++
CXXFLAGS = --c++17 -I$(INCLUDE_DIR) -Mcudalib=curand

# SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TARGETS := $(notdir $(basename $(SRCS)))
DEPS := $(addsuffix .d, $(TARGETS))

# LDFLAGS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
LDFLAGS = -Wl, -Mcudalib=curand

#################  ####################
APPS = mnist sample
ACCLEVELS = noacc acchost acc

all: lib app

app:
	cd apps && $(MAKE) all && cd ..

lib: acc acchost noacc
SUFFIX_ACC = acc
LIB_DIR_ACC = $(LIB_DIR)/$(SUFFIX_ACC)
OBJS_ACC := $(addsuffix .o, $(addprefix $(LIB_DIR_ACC)/, $(TARGETS)))

acc: $(OBJS_ACC)

$(OBJS_ACC):$(LIB_DIR_ACC)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(LIB_DIR_ACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS) -acc -Minfo

SUFFIX_ACCHOST = acchost
LIB_DIR_ACCHOST = $(LIB_DIR)/$(SUFFIX_ACCHOST)
OBJS_ACCHOST := $(addsuffix .o, $(addprefix $(LIB_DIR_ACCHOST)/, $(TARGETS)))

acchost: $(OBJS_ACCHOST)

$(OBJS_ACCHOST):$(LIB_DIR_ACCHOST)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(LIB_DIR_ACCHOST)
	$(CXX) -c $< -o $@ $(CXXFLAGS) -acc=host -Minfo

SUFFIX_NOACC = noacc
LIB_DIR_NOACC = $(LIB_DIR)/$(SUFFIX_NOACC)
OBJS_NOACC := $(addsuffix .o, $(addprefix $(LIB_DIR_NOACC)/, $(TARGETS)))

noacc: $(OBJS_NOACC)

$(OBJS_NOACC):$(LIB_DIR_NOACC)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(LIB_DIR_NOACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(DEPS):%.d:$(SRC_DIR)/%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -c $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*, $(LIB_DIR_ACC)/\1.o $(LIB_DIR_ACCHOST)/\1.o $(LIB_DIR_NOACC)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

.PHONY : clean
clean:
	rm -rf lib

include $(DEPS)
# clean_all: clean
# 	cd mnist_app && $(MAKE) clean
# 	cd sample_app && $(MAKE) clean
################# / ####################

