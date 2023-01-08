CXX = nvc++
CXXFLAGS = --c++17 -I$(INCLUDE_DIR)
LDFLAGS = -Mcudalib
INCLUDE_DIR = ../../src/include
LIB_DIR = ../../lib
BUILD_DIR = build
LIBS = layer network tensor ops utils
TARGETS = training_sample
DEPS := $(TARGETS:%=%.d)
PROGRAM = sample

SUFFIX_ACC = acc
OBJS_ACC = $(TARGETS:%=$(BUILD_DIR)/$(SUFFIX_ACC)/%.o)
LIBS_ACC = $(LIBS:%=$(LIB_DIR)/$(SUFFIX_ACC)/%.o)
FLAGS_ACC = -acc -Minfo

SUFFIX_ACCHOST = acchost
OBJS_ACCHOST = $(TARGETS:%=$(BUILD_DIR)/$(SUFFIX_ACCHOST)/%.o)
LIBS_ACCHOST = $(LIBS:%=$(LIB_DIR)/$(SUFFIX_ACCHOST)/%.o)
FLAGS_ACCHOST = -acc=host -Minfo

SUFFIX_NOACC = noacc
OBJS_NOACC = $(TARGETS:%=$(BUILD_DIR)/$(SUFFIX_NOACC)/%.o)
LIBS_NOACC = $(LIBS:%=$(LIB_DIR)/$(SUFFIX_NOACC)/%.o)
FLAGS_NOACC = 

all: $(SUFFIX_ACC) $(SUFFIX_ACCHOST) $(SUFFIX_NOACC)

$(SUFFIX_ACC) $(SUFFIX_ACCHOST) $(SUFFIX_NOACC):%:$(BUILD_DIR)/$(PROGRAM)_%

##acc
$(BUILD_DIR)/$(PROGRAM)_$(SUFFIX_ACC): $(OBJS_ACC) $(LIBS_ACC)
	$(CXX) -o $@ $(LDFLAGS) $^ $(FLAGS_ACC)

$(OBJS_ACC):$(BUILD_DIR)/$(SUFFIX_ACC)/%.o:%.cpp
	@mkdir -p $(BUILD_DIR)/$(SUFFIX_ACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_ACC)

##acchost
$(BUILD_DIR)/$(PROGRAM)_$(SUFFIX_ACCHOST): $(OBJS_ACCHOST) $(LIBS_ACCHOST)
	$(CXX) -o $@ $(LDFLAGS) $^ $(FLAGS_ACCHOST)

$(OBJS_ACCHOST):$(BUILD_DIR)/$(SUFFIX_ACCHOST)/%.o:%.cpp
	@mkdir -p $(BUILD_DIR)/$(SUFFIX_ACCHOST)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_ACCHOST)

##noacc
$(BUILD_DIR)/$(PROGRAM)_$(SUFFIX_NOACC): $(OBJS_NOACC) $(LIBS_NOACC)
	$(CXX) -o $@ $(LDFLAGS) $^ $(FLAGS_NOACC)

$(OBJS_NOACC):$(BUILD_DIR)/$(SUFFIX_NOACC)/%.o:%.cpp
	@mkdir -p $(BUILD_DIR)/$(SUFFIX_NOACC)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS_NOACC)

$(DEPS):%.d:%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -c $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*, $(BUILD_DIR)/$(SUFFIX_ACC)/\1.o $(BUILD_DIR)/$(SUFFIX_ACCHOST)/\1.o $(BUILD_DIR)/$(SUFFIX_NOACC)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

clean:
	rm -rf build

include $(DEPS)