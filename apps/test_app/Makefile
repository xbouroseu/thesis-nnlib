CXX = nvc++

all: test_acc

test_acc: test.cpp
	@mkdir -p build
	$(CXX) -o build/$@ -acc -Minfo test.cpp

clean:
	rm -rf build