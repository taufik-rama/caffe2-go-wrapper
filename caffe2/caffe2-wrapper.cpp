/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#undef CAFFE2_USE_MKL

#include <string>
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

    caffe2::NetDef net_graph;
    caffe2::Workspace workspace("workspace");
    bool initialized = false;

    int load_model(const char *path) {

        if (!initialized) {

            if(access(path, F_OK) != 0) {
                return -1;
            }

            auto db = std::unique_ptr<caffe2::db::DBReader>(new caffe2::db::DBReader("minidb", path));

            auto meta_net = caffe2::predictor_utils::runGlobalInitialization(std::move(db), &workspace);
            const auto net_init = caffe2::predictor_utils::getNet(
                *meta_net.get(),
                caffe2::PredictorConsts::default_instance().predict_init_net_type());

            if(!workspace.RunNetOnce(net_init)) {
                return -1;
            }

            auto net = caffe2::NetDef(caffe2::predictor_utils::getNet(
                *meta_net.get(),
                caffe2::PredictorConsts::default_instance().predict_net_type()));

            CAFFE_ENFORCE(workspace.CreateNet(net));

            net_graph = net;
            initialized = true;
        }

        return 0;
    }

    int predict(const char *query, struct predict_result *result, int result_size) {

        // Pre-process: tokenize input doc
        std::vector<std::string> tokens;
        std::string docCopy = query;
        tokenize(tokens, docCopy);

        // Feed input to model as tensors
        caffe2::Tensor tensor_val = caffe2::TensorCPUFromValues<std::string>(
            {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens});
        BlobGetMutableTensor(workspace.CreateBlob("tokens_vals_str:value"), caffe2::CPU)
            ->CopyFrom(tensor_val);
        caffe2::Tensor tensor_lens = caffe2::TensorCPUFromValues<int>(
            {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())});
        BlobGetMutableTensor(workspace.CreateBlob("tokens_lens"), caffe2::CPU)
            ->CopyFrom(tensor_lens);

        // Run the model
        if(!workspace.RunNet(net_graph.name())) {
            return -1;
        }

        // Extract and populate results into the response
        for (int i = 0; i < result_size; i++) {

            // If the requested size is bigger than available data, pad it with null terminator
            if(i >= net_graph.external_output().size()) {
                for(int j = 0; j < (result_size - i); j++) {
                    result[i + j].label[0] = '\0';
                }
                break;
            }

            std::string label = net_graph.external_output()[i];
            if(label.length() > 32) {
                strncpy(result[i].label, label.c_str(), 32);
                result[i].label[32] = '\0';
            } else {
                strncpy(result[i].label, label.c_str(), label.length());
                result[i].label[label.length()] = '\0';
            }


            const caffe2::Tensor *tensor_scores = &workspace.GetBlob(label)->Get<caffe2::Tensor>();
            for (int j = 0; j < tensor_scores->numel(); j++) {

                if(j >= 8) {
                    break;
                }

                float score = tensor_scores->data<float>()[j];
                result[i].prob[j] = score;
            }
        }

        return 0;
    }
}
