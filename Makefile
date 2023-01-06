CXX = nvc++
SRCDIR = src
BINDIR = bin
BUILDDIR = build
CPPFLAGS = --c++17
CPPINCLUDES = -I/usr/local/include/opencv4 -Isrc/libraries -Mcudalib=curand
LINKLIBS = -Wl,-lopencv_core,-lopencv_imgcodecs,-lopencv_highgui,-lopencv_imgproc -Mcudalib=curand
SRCFILES = training mnist
LIBFILES = layer network tensor ops utils

##### noacc
BINDIR = bin/noacc
ACCFLAGS = 

$(BUILDDIR)/app_noacc: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS)

${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/libraries/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#######

##### acc=host
BINDIR = bin/acc_host
ACCFLAGS = -acc=host -Minfo

$(BUILDDIR)/app_acc_host: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS)

${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/libraries/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#######

##### noacc
BINDIR = bin/acc
ACCFLAGS = -acc -Minfo

$(BUILDDIR)/app_acc: ${foreach trg,${SRCFILES} ${LIBFILES},${BINDIR}/${trg}.o}
	$(CXX) -o $@ $^ $(LINKLIBS)

${foreach trg, ${SRCFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}

${foreach trg, ${LIBFILES}, ${BINDIR}/${trg}.o} : $(BINDIR)/%.o : $(SRCDIR)/libraries/%.cpp
	${CXX} -c $^ -o $@ ${CPPFLAGS} ${CPPINCLUDES} ${ACCFLAGS}
#######