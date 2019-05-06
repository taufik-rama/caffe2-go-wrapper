/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "caffe2-wrapper-types.hpp"

extern "C" {

    /**
     * Initialize the `predictor` with the model located on `path`
     * 
     * Returns status indication successful process (0 for success, fails otherwise)
     */
    int cf2_load_model(
        struct cf2_predictor *predictor, 
        const char *path
    );

    /**
     * Predict a given keyword
     * 
     * `in` : The actual keyword to predict
     * `out`: An array of `cf2_predictor_result` that will contains
     *        the result
     * 
     * Returns 0 on success
     */
    int cf2_predict(
        struct cf2_predictor *predictor, 
        const char *in, 
        struct cf2_predictor_result out[PREDICT_RESULT_SIZE]
    );
}