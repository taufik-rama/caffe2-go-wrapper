CPPFLAGS += -g3 -std=c++11 -Iinclude -Iinclude-pytorch -Iinclude-mkl

libcaffe2-wrapper.a: libcaffe2-wrapper.o
	ar rcs lib/$@ *.o lib/*.so

libcaffe2-wrapper.o: caffe2-wrapper.cpp
	g++ -c $(CPPFLAGS) $^ -Wl,-rpath=lib -o $@

.PHONY: clean

clean:
	rm -f *.o