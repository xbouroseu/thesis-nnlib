CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -I../include
LIBDIR = ../lib
BINDIR = bin
LINKLIBS = -Mcudalib
LIBS = layer network tensor ops utils

all: mnist_acc mnist_acchost mnist_noacc

mnist_acc: $(BINDIR)/training_acc.o $(BINDIR)/mnist_acc.o
	$(CXX) -o $@ $^ $(LINKLIBS) -acc $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)

mnist_acchost: $(BINDIR)/training_acchost.o $(BINDIR)/mnist_acchost.o
	$(CXX) -o $@ $^ $(LINKLIBS) -acc=host $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acchost.o)

mnist_noacc: $(BINDIR)/training_noacc.o $(BINDIR)/mnist_noacc.o
	$(CXX) -o $@ $^ $(LINKLIBS) $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_noacc.o)

$(BINDIR)/training_acc.o $(BINDIR)/mnist_acc.o: $(BINDIR)/%_acc.o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

$(BINDIR)/training_acchost.o $(BINDIR)/mnist_acchost.o: $(BINDIR)/%_acchost.o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc=host -Minfo

$(BINDIR)/training_noacc.o $(BINDIR)/mnist_noacc.o: $(BINDIR)/%_noacc.o: %.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES)