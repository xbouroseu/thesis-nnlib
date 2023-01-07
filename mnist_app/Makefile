CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -I../include
LIBDIR = ../lib
BINDIR = bin
BUILDDIR = build
LINKLIBS = -Mcudalib
LIBS = layer network tensor ops utils
SRCFILES = training.cpp mnist.cpp mnist.hpp

$(BUILDDIR)/mnist_acc: $(BUILDDIR)/mnist_%: $(BINDIR)/training_%.o $(BINDIR)/mnist_%.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)
	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

$(BINDIR)/training_acc.o $(BINDIR)/mnist_acc.o: $(BINDIR)/%_acc.o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

# build/mnist_acc_test: bin/mnist_acc.o bin/training_acc.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)
# 	$(CXX) -o $@ $(LINKLIBS) -acc -Minfo $^

# bin/training_acc.o bin/mnist_acc.o: bin/%_acc.o : %.cpp
# 	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

# all: mnist_acc mnist_acchost mnist_noacc


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