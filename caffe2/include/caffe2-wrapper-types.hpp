/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * This struct will act as a return value type for the prediction
 */
struct cf2_predict_result {
    
    // `label` is the predicted label
    char *label;

    // `prob` is an array of label prediction probability values
    double prob[8];

};