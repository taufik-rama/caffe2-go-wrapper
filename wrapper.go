package main

// #cgo CPPFLAGS: -Icaffe2/include/
// #cgo LDFLAGS: -L${SRCDIR}/caffe2/lib -lcaffe2-wrapper -lc10 -lcaffe2 -lpthread -lprotobuf -lstdc++
// #cgo LDFLAGS: -Wl,-rpath=${SRCDIR}/caffe2/lib
// #include <stdlib.h>
// #include <caffe2-wrapper-types.hpp>
// int load_model(const char *path);
// int predict(const char *query_in, struct predict_result *result, int result_size);
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Model uses Caffe2 for it's prediction
type Model struct {
	isInitialized bool
}

// New should be used to instantiate the model.
// FastTest needs some initialization for the model binary located on `file`.
func New(file string) (*Model, error) {

	status := C.load_model(C.CString(file))

	if status != 0 {
		return nil, fmt.Errorf("Cannot initialize model on `%s`", file)
	}

	return &Model{
		isInitialized: true,
	}, nil
}

// Predict the `keyword`
func (m *Model) Predict(keyword string) error {

	if !m.isInitialized {
		return errors.New("The Caffe2 model needs to be initialized first. It's should be done inside the `New()` function")
	}

	resultSize := 16
	result := make([]C.struct_predict_result, resultSize)

	resultLabelSize := 32
	for i := 0; i < resultSize; i++ {
		result[i].label = (*C.char)(C.malloc(C.ulong(resultLabelSize)))
	}

	status := C.predict(
		C.CString(keyword),
		&result[0],
		C.int(resultSize),
	)

	if status != 0 {
		return fmt.Errorf("Exception when predicting `%s`", keyword)
	}

	// Here's the result from C
	for i := 0; i < resultSize; i++ {
		resultLabel := C.GoString(result[i].label)
		resultProb := result[i].prob
		fmt.Println(resultLabel, resultProb)
	}

	for i := 0; i < resultSize; i++ {
		C.free(unsafe.Pointer(result[i].label))
	}

	return nil
}
