CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -I../include
LIBDIR = ../lib
BINDIR = bin
BUILDDIR = build
LINKLIBS = -Mcudalib
LIBS = layer network tensor ops utils
SRCFILES = training.cpp mnist.cpp mnist.hpp
TARGETS = training mnist

all: $(BUILDDIR)/mnist_acc $(BUILDDIR)/mnist_acchost $(BUILDDIR)/mnist_noacc

SUFFIX_ACC = acc
$(BUILDDIR)/mnist_$(SUFFIX_ACC): $(BUILDDIR)/mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_%.o)
	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

$(foreach trg, $(TARGETS), $(BINDIR)/$(trg)_$(SUFFIX_ACC).o): $(BINDIR)/%_$(SUFFIX_ACC).o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

SUFFIX_ACCHOST = acchost
$(BUILDDIR)/mnist_$(SUFFIX_ACCHOST): $(BUILDDIR)/mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_%.o)
	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

$(foreach trg, $(TARGETS), $(BINDIR)/$(trg)_$(SUFFIX_ACCHOST).o): $(BINDIR)/%_$(SUFFIX_ACCHOST).o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc=host -Minfo

SUFFIX_NOACC = noacc
$(BUILDDIR)/mnist_$(SUFFIX_NOACC): $(BUILDDIR)/mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_%.o)
	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

$(foreach trg, $(TARGETS), $(BINDIR)/$(trg)_$(SUFFIX_NOACC).o): $(BINDIR)/%_$(SUFFIX_NOACC).o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

# build/mnist_acc_test: bin/mnist_acc.o bin/training_acc.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)
# 	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

# bin/training_acc.o bin/mnist_acc.o: bin/%_acc.o : %.cpp
# 	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

# all: 


# mnist_acchost: mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acchost.o) 
# 	$(CXX) -o $@ $(LINKLIBS) -acc=host -Minfo $^

# mnist_noacc: mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_noacc.o)
# 	$(CXX) -o $@ $(LINKLIBS) $^



# $(BINDIR)/training_acchost.o $(BINDIR)/mnist_acchost.o: $(BINDIR)/%_acchost.o: %.cpp
# 	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc=host -Minfo

# $(BINDIR)/training_noacc.o $(BINDIR)/mnist_noacc.o: $(BINDIR)/%_noacc.o: %.cpp
# 	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES)

clean:
	rm bin/* build/*