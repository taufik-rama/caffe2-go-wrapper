# Caffe2 Golang Wrapper

## Overview

Here's my attempt at wrapping Caffe2 C++ library with Golang CGO.

Note: Caffe2 is now a part of pytorch library, so need to compile it from pytorch repository

## Requirements

- `git`
- `make`
- And other requirements for the Caffe2/Pytorch library & Intel MKL-DNN library.

## Compiling

- Clone the `pytorch` [git repository](https://github.com/pytorch/pytorch) & compile it.

- Clone the `mkl-dnn` [git repository](https://github.com/intel/mkl-dnn) & get the external header files.

- Clone this repository.

    ```Bash
    $ git clone https://github.com/taufik-rama/caffe2-go-binding
    # Cloning...
    ```

- Copy all of the `torch` package include file (should be on `torch/include/`) into `caffe2/include-pytorch/`

    ```bash
    # On pytorch directory after compilation
    $ cp -r torch/include/* /path/to/caffe2/include-pytorch/
    ```

- Copy `libc10.so` & `libcaffe2.so` from `pytorch` build result (should be on `build/lib`) into `caffe2/lib/` directory.

    ```bash
    # On pytorch directory after compilation
    $ cp build/lib/libc10.so build/lib/libcaffe2.so /path/to/caffe2/lib/
    ```

- Copy all of the `mkl-dnn` external header file (should be on `external/mklxx/include/`) into `caffe2/include-mkl`

    ```bash
    # On pytorch directory after compilation
    $ cp external/mkl{xx}/include/* /path/to/caffe2/include-mkl/
    ```

- Compile the C project

    ```Bash
    $ cd caffe2 && make
    # Compiling...
    ```

- Build the Go package normally

    ```Bash
    $ go build
    # Compiling...
    ```