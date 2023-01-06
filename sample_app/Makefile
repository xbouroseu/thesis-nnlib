CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -I../include
LIBDIR = ../lib
BINDIR = bin
LINKLIBS = -Mcudalib
LIBS = layer network tensor ops utils

all: sample_noacc sample_acchost sample_acc

sample_acc: $(BINDIR)/training_sample_acc.o
	$(CXX) -o $@ $^ $(LINKLIBS) -acc $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)

sample_acchost: $(BINDIR)/training_sample_acchost.o
	$(CXX) -o $@ $^ $(LINKLIBS) -acc=host $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acchost.o)

sample_noacc: $(BINDIR)/training_sample_noacc.o
	$(CXX) -o $@ $^ $(LINKLIBS) $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_noacc.o)

$(BINDIR)/training_sample_noacc.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES)

$(BINDIR)/training_sample_acchost.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc=host -Minfo

$(BINDIR)/training_sample_acc.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo
