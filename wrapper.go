package main

// #cgo CPPFLAGS: -Icaffe2/include/
// #cgo LDFLAGS: -L${SRCDIR}/caffe2/lib -lcaffe2-wrapper -lc10 -lcaffe2 -lpthread -lprotobuf -lstdc++
// #cgo LDFLAGS: -Wl,-rpath=${SRCDIR}/caffe2/lib
//
// #include <stdlib.h>
//
// // should also be changed on `caffe2-wrapper` source code
// // this is here to help the developer use the constants on Go
// #define PREDICT_RESULT_SIZE 16
// #define LABEL_SIZE 32
// #define PROB_SIZE 8
//
// struct cf2_predictor_result {
//
//     // the string output from caffe2 model
//     char label[LABEL_SIZE];
//
//     // array of prediction values of the `label`
//     double prob[PROB_SIZE];
// };
//
// int cf2_create(
// 	   const char *path
// );
//
// int cf2_predict(
// 	   const char *in, 
// 	   struct cf2_predictor_result out[PREDICT_RESULT_SIZE],
// 	   const int predictor_index
// );
import "C"

import (
	"errors"
	"fmt"
)

// Model uses Caffe2 for it's prediction
type Model struct {
	id int
	isInitialized bool
}

// New should be used to instantiate the model.
// Caffe2 needs some initialization for the model binary located on `file`.
func New(file string) (*Model, error) {

	predictorID := C.cf2_create(C.CString(file))

	if predictorID == -1 {
		return nil, fmt.Errorf("Cannot initialize model on `%s`", file)
	}

	return &Model{
		id:     int(predictorID),
		isInitialized: true,
	}, nil
}

// Predict the `keyword`
func (m *Model) Predict(keyword string) error {

	if !m.isInitialized {
		return errors.New("The Caffe2 model needs to be initialized first. It's should be done inside the `New()` function")
	}

	result := make([]C.struct_cf2_predictor_result, C.PREDICT_RESULT_SIZE)

	status := C.cf2_predict(
		C.CString(keyword),
		&result[0],
		C.int(m.id),
	)

	if status != 0 {
		return fmt.Errorf("Exception when predicting `%s`", keyword)
	}

	// Here's the result from C
	for i := 0; i < C.PREDICT_RESULT_SIZE; i++ {
		resultLabel := C.GoString(&result[i].label[0])
		resultProb := result[i].prob
		fmt.Println(resultLabel, resultProb)
	}

	return nil
}
