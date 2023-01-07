CXX = nvc++

all: test_acc

test_acc: test.cpp
	$(CXX) -o test_$@ -acc -Minfo test.cpp

clean:
	rm test_acc