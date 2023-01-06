CXX = nvc++
SRCDIR = src
BINDIR = bin
BUILDDIR = build
CPPFLAGS = --c++17 -Mcudalib=curand
CPPINCLUDES = 
# LINKLIBS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
LINKLIBS = -Wl, -Mcudalib=curand

#################  ####################
LIB_FILES = layer network tensor ops utils
MNIST_FILES = training mnist
SAMPLE_FILES = training_sample

LIB_SRCDIR = $(SRCDIR)/libraries
MNIST_SRCDIR = $(SRCDIR)/mnist
SAMPLE_SRCDIR = $(SRCDIR)/sample

LIB_BINDIR = $(BUILDDIR)/lib
MNIST_LIBDIR = $(BUILDDIR)/mnist
SAMPLE_LIBDIR = $(BUILDDIR)/sample

#### noacc
ACCSUFFIX = noacc
BINDIR = ${BUILDDIR}/${ACCSUFFIX}
ACCFLAGS = 

$(BUILDDIR)/mnist_${ACCSUFFIX}: ${foreach trg,${MNISTFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS)

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : /%.o : $(LIBDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${MNISTFILES}, ${BINDIR}/${trg}.o} : /%.o : $(MNISTDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${SAMPLEFILES}, ${BINDIR}/${trg}.o} : /%.o : $(SAMPLEDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#### acc=host
BINDIR = ${BUILDDIR}/acc_host
ACCFLAGS = -acc=host -Minfo
${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : /%.o : $(FILESDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${MNISTFILES}, ${BINDIR}/${trg}.o} : /%.o : $(FILESDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#### acc
BINDIR = ${BUILDDIR}/noacc
ACCFLAGS = -acc -Minfo
${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : /%.o : $(FILESDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${MNISTFILES}, ${BINDIR}/${trg}.o} : /%.o : $(FILESDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
################# / ####################

################# MNIST ####################

FILESDIR = $(SRCDIR)/mnist

#### noacc
BINDIR = ${BUILDDIR}/noacc
ACCFLAGS = 


#### acchost
BINDIR = ${BUILDDIR}/acc_host
ACCFLAGS = -acc=host -Minfo

${foreach trg, ${MNISTFILES}, ${BINDIR}/${trg}.o}  : $(BINDIR)/%.o : $(FILESDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

#### acc
BINDIR = ${BUILDDIR}/acc
ACCFLAGS = -acc -Minfo

${foreach trg, ${MNISTFILES}, ${BINDIR}/${trg}.o}  : $(BINDIR)/%.o : $(FILESDIR)/%.cpp
		${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
################# /MNIST ####################

################# APPS ######################
#### MNIST

$(BUILDDIR)/mnist_noacc: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}

##### noacc


${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

#######

##### acc=host
BINDIR = bin/acc_host
ACCFLAGS = -acc=host -Minfo

$(BUILDDIR)/app_acc_host: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS) -acc=host

${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/libraries/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#######

##### noacc
BINDIR = bin/acc
ACCFLAGS = -acc -Minfo

$(BUILDDIR)/app_acc: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS) -acc

${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/libraries/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#######