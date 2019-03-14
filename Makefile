# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

CPPFLAGS += -g -std=c++11 -std=c++14 \
  -I./gen-cpp \
  -I/usr/local/pytorch -I/usr/local/pytorch/build \
	-I/usr/local/pytorch/aten/src/ \
	-I/usr/local/pytorch/third_party/protobuf/src/
PREDICTOR_LDFLAGS += -L/usr/local/pytorch/build/lib \
	-L/usr/local/lib \
  -lpthread -lcaffe2 -lprotobuf -lc10 -lcurl

predictor-cli: predictor-cli.o
	g++ $^ $(PREDICTOR_LDFLAGS) -o $@

clean:
	rm -f *.o predictor-cli
