CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -I../include
LIBDIR = ../lib
BINDIR = bin
LINKLIBS = -Mcudalib
LIBS = layer network tensor ops utils

all: sample_noacc sample_acchost sample_acc

sample_acc: $(BINDIR)/training_sample_acc.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acc.o)
	$(CXX) -o $@ $^ $(LINKLIBS) -acc 

sample_acchost: $(BINDIR)/training_sample_acchost.o $(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_acchost.o)
	$(CXX) -o $@ $^ $(LINKLIBS) -acc=host 

sample_noacc: $(BINDIR)/training_sample_noacc.o$(foreach lb, $(LIBS), $(LIBDIR)/$(lb)_noacc.o)
	$(CXX) -o $@ $^ $(LINKLIBS) 

$(BINDIR)/training_sample_noacc.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES)

$(BINDIR)/training_sample_acchost.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc=host -Minfo

$(BINDIR)/training_sample_acc.o: training_sample.cpp
	$(CXX) -c $^ -o $@ $(CPPFLAGS) $(CPPINCLUDES) -acc -Minfo

clean:
	rm bin/* sample_acc sample_acchost sample_noacc
