CPPFLAGS += -std=c++11 -Iinclude -Iinclude-pytorch -Iinclude-mkl

libcaffe2-wrapper.a: libcaffe2-wrapper.o
	ar rcs -o lib/$@ lib/*.o

libcaffe2-wrapper.o: caffe2-wrapper.cpp
	g++ -c -fPIC $(CPPFLAGS) $^ -o lib/$@

.PHONY: clean

clean:
	rm -f lib/libcaffe2-wrapper.o lib/libcaffe2-wrapper.a lib/caffe2

# Use this to build a binary with this library
# I use this to test the C/C++ code
bin: caffe2-wrapper.cpp
	g++ -fPIC $(CPPFLAGS) $^ \
	-Llib/ -lc10 -lcaffe2 -lpthread -lprotobuf -lstdc++ \
	-Wl,-rpath=lib/ \
	-o lib/caffe2