CPPFLAGS += -std=c++11 -Iinclude -Iinclude-pytorch -Iinclude-mkl

libcaffe2-wrapper.so: caffe2-wrapper.cpp
	g++ -shared -fPIC $(CPPFLAGS) lib/*.so $^ -o lib/$@

.PHONY: clean

clean:
	rm -f lib/libcaffe2-wrapper.so