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

all:  noacc acchost acc

#### noacc
noacc: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_noacc.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_noacc.o} : $(BINDIR)/%_noacc.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES}

#### acchost
acchost: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acchost.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acchost.o} : $(BINDIR)/%_acchost.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} -acc=host -Minfo

#### acc
acc: ${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acc.o}

${foreach trg, ${TARGET_NAMES}, ${BINDIR}/${trg}_acc.o} : $(BINDIR)/%_acc.o: $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} -acc -Minfo

################# / ####################