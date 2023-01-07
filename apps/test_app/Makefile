CXX = nvc++

all: test_acc

test_acc: test.cpp
	$(CXX) -o build/$@ -acc -Minfo test.cpp

clean:
	rm build/test_acc