/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

extern "C" {

    /**
     * Initialize the caffe2 model located on `path`
     * returns 0 on success
     */
    int load_model(const char *path);

    /**
     * Predict a given keyword
     * `query_in`: The actual keyword to predict
     * `result`: An array of `predict_result` that will contain the result
     * `result_size`: The allocation size of `result`
     * returns 0 on success
     */
    int predict(const char *query_in, struct predict_result *result, int result_size);
}