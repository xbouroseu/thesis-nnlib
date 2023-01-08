CXX = nvc++
CXXFLAGS = --c++17 -I$(INCLUDE_DIR) -Mcudalib=curand
LDFLAGS = -Wl, -Mcudalib=curand
# LDFLAGS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
SRC_DIR = src
INCLUDE_DIR = $(SRC_DIR)/include
BUILD_DIR = lib
# SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TARGETS := $(notdir $(basename $(SRCS)))
DEPS := $(addsuffix .d, $(TARGETS))
#################  ####################

all: lib app

lib: acc acchost noacc

app:
	@cd apps && $(MAKE) all && cd ..

SUFFIX_ACC = acc
BUILD_DIR_ACC = $(BUILD_DIR)/$(SUFFIX_ACC)
OBJS_ACC := $(addsuffix .o, $(addprefix $(BUILD_DIR_ACC)/, $(TARGETS)))
FLAGS_ACC = -acc -Minfo

SUFFIX_ACCHOST = acchost
BUILD_DIR_ACCHOST = $(BUILD_DIR)/$(SUFFIX_ACCHOST)
OBJS_ACCHOST := $(addsuffix .o, $(addprefix $(BUILD_DIR_ACCHOST)/, $(TARGETS)))
FLAGS_ACCHOST = -acc=host -Minfo

SUFFIX_NOACC = noacc
BUILD_DIR_NOACC = $(BUILD_DIR)/$(SUFFIX_NOACC)
OBJS_NOACC := $(addsuffix .o, $(addprefix $(BUILD_DIR_NOACC)/, $(TARGETS)))
FLAGS_NOACC = 

$(SUFFIX_ACC): $(OBJS_ACC)
$(SUFFIX_ACCHOST): $(OBJS_ACCHOST)
$(SUFFIX_NOACC): $(OBJS_NOACC)

##acc
$(OBJS_ACC):$(BUILD_DIR_ACC)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR_ACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_ACC)

##acchost
$(OBJS_ACCHOST):$(BUILD_DIR_ACCHOST)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR_ACCHOST)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_ACCHOST)

##noacc
$(OBJS_NOACC):$(BUILD_DIR_NOACC)/%.o:$(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR_NOACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_NOACC)

$(DEPS):%.d:$(SRC_DIR)/%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -c $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*, $(BUILD_DIR_ACC)/\1.o $(BUILD_DIR_ACCHOST)/\1.o $(BUILD_DIR_NOACC)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

.PHONY : clean
clean:
	rm -rf lib

include $(DEPS)
# clean_all: clean
# 	cd mnist_app && $(MAKE) clean
# 	cd sample_app && $(MAKE) clean
################# / ####################

