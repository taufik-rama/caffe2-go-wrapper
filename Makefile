# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

CPPFLAGS += -g3 -std=c++11 \
  -I./gen-cpp \
  -I/usr/local/pytorch -I/usr/local/pytorch/build \
	-I/usr/local/pytorch/aten/src/ \
	-I/usr/local/pytorch/third_party/protobuf/src/ \
	-I./pytorch/torch/lib/include/ \
	-I/home/nakama/anaconda3/include/
PREDICTOR_LDFLAGS += -L/usr/local/pytorch/build/lib \
	-L/usr/local/lib \
	-L. \
  -lpthread -lcaffe2 -lprotobuf -lc10 -lcurl

predictor-cli: predictor-cli.o
	g++ $^ $(PREDICTOR_LDFLAGS) -o $@

clean:
	rm -f *.o predictor-cli
