/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "caffe2-wrapper-types.hpp"

extern "C" {

    /**
     * Initialize the `predictor` with the model located on `path`.
     * 
     * Returns the id / index of the predictor, -1 on error.
     */
    int cf2_create(
        const char *path
    );

    /**
     * Predict a given keyword
     * 
     * `in` : The actual keyword to predict
     * `out`: An array of `cf2_predictor_result` that will contains
     *        the result
     * `predictor_index`: The index of the allocated predictor(s) based on
     *                    `cf2_initialize()` call
     * 
     * Returns 0 on success
     */
    int cf2_predict(
        const char *in, 
        struct cf2_predictor_result out[PREDICT_RESULT_SIZE],
        const int predictor_index
    );
}