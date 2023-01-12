SUBDIRS := $(filter-out ., $(shell find . -type d))

all: mnist sample test legacy

mnist: ./mnist_app
	cd mnist_app && $(MAKE) all && cd

sample: ./sample_app
	cd sample_app && $(MAKE) all && cd

test: ./test_app
	cd test_app && $(MAKE) all && cd

legacy: ./test_app
	cd legacy_app && $(MAKE) all && cd