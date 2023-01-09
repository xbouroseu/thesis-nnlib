CXX = nvc++

all: build/test_acc

build/test_acc: test.cpp
	@mkdir -p build
	$(CXX) -o $@ -acc -Minfo test.cpp

clean:
	rm -rf build