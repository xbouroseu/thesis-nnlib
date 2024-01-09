SUBDIRS := $(filter-out ., $(shell find . -type d))

all: mnist
mnist: ./mnist_app
	cd mnist_app && $(MAKE) all && cd