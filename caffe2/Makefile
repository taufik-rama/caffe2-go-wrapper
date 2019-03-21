CPPFLAGS += -std=c++11 -Iinclude -Iinclude-pytorch -Iinclude-mkl

libcaffe2-wrapper.a: libcaffe2-wrapper.o
	ar rcs -o lib/$@ lib/*.o

libcaffe2-wrapper.o: caffe2-wrapper.cpp
	g++ -c -fPIC $(CPPFLAGS) $^ -o lib/$@

.PHONY: clean

clean:
	rm -f lib/libcaffe2-wrapper.o lib/libcaffe2-wrapper.a