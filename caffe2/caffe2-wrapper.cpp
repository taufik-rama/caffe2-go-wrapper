/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#undef CAFFE2_USE_MKL

#include <string>
#include <pthread.h>
#include "caffe2-wrapper.hpp"
#include "caffe2-wrapper-types.hpp"
#include "caffe2/predictor/predictor_utils.h"

void tokenize(std::vector<std::string> &result, std::string &in) {
        
    std::transform(in.begin(), in.end(), in.begin(), ::tolower);

    int start = 0;
    int end = 0;
    for (int i = 0; i < in.length(); i++) {
        if (isspace(in.at(i))) {
            end = i;
            if (end != start) {
                result.push_back(in.substr(start, end - start));
            }

            start = i + 1;
        }
    }

    if (start < in.length()) {
        result.push_back(in.substr(start, in.length() - start));
    }

    if (result.size() == 0) {

        // Add PAD_TOKEN in case of empty text
        result.push_back("<pad>");
    }
}

extern "C" {

    pthread_mutex_t predictors_lock = PTHREAD_MUTEX_INITIALIZER;
    std::vector<cf2_predictor> predictors;

    int cf2_create(
        const char *path
    ) {

        // check file access
        if (access(path, F_OK) != 0) {
            return -1;
        }

        cf2_predictor p;
        std::shared_ptr<caffe2::Workspace> workspace(new caffe2::Workspace("workspace"));
        p.workspace_ref = std::move(workspace);

        auto db = std::unique_ptr<caffe2::db::DBReader>(new caffe2::db::DBReader("minidb", path));
        auto meta_net = caffe2::predictor_utils::runGlobalInitialization(std::move(db), &(*p.workspace_ref));
        const auto net_init = caffe2::predictor_utils::getNet(
            *meta_net.get(),
            caffe2::PredictorConsts::default_instance().predict_init_net_type());
        if (!p.workspace_ref->RunNetOnce(net_init)) {
            return -1;
        }

        std::shared_ptr<caffe2::NetDef> net_ref(new caffe2::NetDef(
            caffe2::predictor_utils::getNet(
                *meta_net.get(),
                caffe2::PredictorConsts::default_instance().predict_net_type()
            )
        ));
        CAFFE_ENFORCE(p.workspace_ref->CreateNet(*net_ref));

        // value that'll be used when we predict
        p.net_graph_ref = std::move(net_ref);
        p.mutex = PTHREAD_MUTEX_INITIALIZER;

        pthread_mutex_lock(&predictors_lock);
        predictors.push_back(p);
        int last_size = predictors.size();
        pthread_mutex_unlock(&predictors_lock);

        // the index is subtracted by 1 (index is zero based)
        return last_size - 1;
    }

    int cf2_predict(
        const char *in, 
        struct cf2_predictor_result out[PREDICT_RESULT_SIZE],
        const int predictor_index
    ) {

        // check the vector bounds
        try {
            predictors.at(predictor_index);
        } catch (const std::out_of_range& e) {
            return -1;
        }

        // pre-process: tokenize input
        std::vector<std::string> tokens;
        std::string docCopy = in;
        tokenize(tokens, docCopy);

        // tensor input
        caffe2::Tensor tensor_val = caffe2::TensorCPUFromValues<std::string>(
            {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens});
        caffe2::Tensor tensor_lens = caffe2::TensorCPUFromValues<int>(
            {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())});

        pthread_mutex_lock(&predictors.at(predictor_index).mutex);

        // output pairs (label -> probability)
        std::vector<std::pair<const std::string, const caffe2::Tensor*>> results;

        {
            cf2_predictor *predictor = &predictors.at(predictor_index);

            // feed input to model as tensors
            BlobGetMutableTensor(predictor->workspace_ref->CreateBlob("tokens_vals_str:value"), caffe2::CPU)
                ->CopyFrom(tensor_val);
            BlobGetMutableTensor(predictor->workspace_ref->CreateBlob("tokens_lens"), caffe2::CPU)
                ->CopyFrom(tensor_lens);

            // run the model
            if (!predictor->workspace_ref->RunNet(predictor->net_graph_ref->name())) {
                pthread_mutex_unlock(&predictor->mutex);
                return -1;
            }

            // get the result from net graph
            int prediction_size = predictor->net_graph_ref->external_output().size();
            for (int i = 0; i < PREDICT_RESULT_SIZE; i++) {

                // if the allocated size is bigger than the model output size,
                // pad it w/ null terminator
                if(i >= prediction_size) {
                    for(int j = 0; j < (PREDICT_RESULT_SIZE - i); j++) {
                        out[i + j].label[0] = '\0';
                    }
                    break;
                }

                auto label = predictor->net_graph_ref->external_output()[i];
                const caffe2::Tensor *tensor_scores = &predictor->workspace_ref->GetBlob(label)->Get<caffe2::Tensor>();

                results.push_back(std::make_pair(label, tensor_scores));
            }
        }

        pthread_mutex_unlock(&predictors.at(predictor_index).mutex);

        // ignore the excessive result
        if (results.size() > PREDICT_RESULT_SIZE) {
            results.resize(PREDICT_RESULT_SIZE);
        }

        // index for `out`
        int i = 0;

        // extract and populate results into `out`
        for (auto iter = results.cbegin(); iter != results.cend(); i++, iter++) {

            auto result = *iter;

            // ignore the rest of the model output label if
            // it exceed the allocated size
            if(result.first.length() > LABEL_SIZE) {
                std::strncpy(out[i].label, result.first.c_str(), LABEL_SIZE);
                out[i].label[LABEL_SIZE] = '\0';
            } else {
                std::strncpy(out[i].label, result.first.c_str(), result.first.length());
                out[i].label[result.first.length()] = '\0';
            }

            int prob_size = result.second->numel();
            for (int j = 0; j < PROB_SIZE; j++) {

                // if the allocated size is bigger than the model output size,
                // pad it w/ 0
                if(j >= prob_size) {
                    for(int k = 0; k < (PROB_SIZE - j); k++) {
                        out[i].prob[j + k] = 0;
                    }
                    break;
                }

                float score = result.second->data<float>()[j];
                out[i].prob[j] = score;
            }
        }

        return 0;
    }
}

// uncomment these if you want to build the binary
// I use it mostly to test the output
// build with `make bin`
// int main() {
//     auto id = cf2_create("../basic-model.c2");
//     struct cf2_predictor_result result[PREDICT_RESULT_SIZE];
    
//     if(cf2_predict("prediction sentence", result, id) == -1) {
//         printf("Error on prediction\n");
//         return -1;
//     }

//     for(int i = 0; i < PREDICT_RESULT_SIZE; i++) {
//         printf("%s ", result[i].label);
//         for(int j = 0; j < PROB_SIZE; j++) {
//             printf("%f ", result[i].prob[j]);
//         }
//         printf("\n");
//     }
// }