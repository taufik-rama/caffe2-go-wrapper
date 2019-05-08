/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdbool.h>
#include <pthread.h>
#include "caffe2/predictor/predictor_utils.h"

/**
 * Prediction size for the `cf2_predict` result array size
 */
#define PREDICT_RESULT_SIZE 16

/**
 * Label size for the caffe2 model output
 */
#define LABEL_SIZE 32

/**
 * Probability size for the caffe2 model output
 */
#define PROB_SIZE 8

/**
 * This struct contains the variable needed by caffe2 library for its 
 * prediction
 */
struct cf2_predictor {

    // each caffe2 workspace should be deleted once all the
    // reference is gone, so we use `shared_ptr` to achive that
    std::shared_ptr<caffe2::Workspace> workspace_ref;

    // caffe2 prediction graph
    std::shared_ptr<caffe2::NetDef> net_graph_ref;

    // prevents race condition during prediction process
    pthread_mutex_t mutex;
};

/**
 * This struct will act as a return value type for the prediction
 */
struct cf2_predictor_result {
    
    // the string output from caffe2 model
    char label[LABEL_SIZE];

    // array of prediction values of the `label`
    double prob[PROB_SIZE];
};