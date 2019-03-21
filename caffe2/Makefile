CPPFLAGS += -g3 -std=c++11 -Iinclude -Iinclude-pytorch -Iinclude-mkl

libcaffe2-wrapper.so: caffe2-wrapper.cpp
	g++ -shared -fPIC $(CPPFLAGS) lib/*.so $^ -Wl,-rpath=lib -o lib/$@

.PHONY: clean

clean:
	rm -f lib/libcaffe2-wrapper.so