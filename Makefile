CXX = nvc++
CPPFLAGS = --c++17
CPPINCLUDES = -Iinclude -Mcudalib=curand

SRCDIR = src
HEADERDIR = include
BINDIR = lib

# LINKLIBS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
LINKLIBS = -Wl, -Mcudalib=curand

#################  ####################
TARGET_NAMES = layer network tensor ops utils

APPS = mnist sample
ACCLEVELS = noacc acchost acc

all: lib
	cd mnist_app && $(MAKE) all
	cd sample_app && $(MAKE) all

lib: $(foreach acclvl, $(ACCLEVELS), lib_$(acclvl))

### todo all this foreach in 1 ^^
#### noacc
lib_noacc: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_noacc.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_noacc.o} : $(BINDIR)/%_noacc.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES}

#### acchost
lib_acchost: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acchost.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acchost.o} : $(BINDIR)/%_acchost.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} -acc=host -Minfo

#### acc
lib_acc: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acc.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acc.o} : $(BINDIR)/%_acc.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} -acc -Minfo

clean:
	rm lib/*

clean_all: clean
	cd mnist_app && $(MAKE) clean
	cd sample_app && $(MAKE) clean
################# / ####################

