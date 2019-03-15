// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <map>

#undef CAFFE2_USE_MKL

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/predictor/predictor_utils.h"
#include "caffe2/proto/predictor_consts.pb.h"
#include "caffe2/utils/proto_utils.h"

// using namespace std;
// using namespace caffe2;
// using namespace db;
// using namespace predictor_utils;

extern "C" {

    caffe2::NetDef net_graph;
    caffe2::Workspace workspace("workspace");
    bool initialized = false;

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

    //       auto db = unique_ptr<DBReader>(new DBReader("minidb", modelFile));
    //       auto metaNetDef = runGlobalInitialization(move(db), &workspace);
    //       const auto predictInitNet = getNet(
    //         *metaNetDef.get(),
    //         PredictorConsts::default_instance().predict_init_net_type()
    //       );
    //       CAFFE_ENFORCE(workspace.RunNetOnce(predictInitNet));

    //       auto predictNet = NetDef(getNet(
    //         *metaNetDef.get(),
    //         PredictorConsts::default_instance().predict_net_type()
    //       ));
    //       CAFFE_ENFORCE(workspace.CreateNet(predictNet));

    //       return predictNet;

    int load_model(const char *path) {

        if (!initialized) {

            if(!access(path, F_OK) != -1) {
                return -1;
            }

            // auto db = std::unique_ptr<caffe2::db::DBReader>(new caffe2::db::DBReader::DBReader("minidb", path));
            auto db = std::unique_ptr<caffe2::db::DBReader>(new caffe2::db::DBReader("minidb", path));

            auto meta_net = caffe2::predictor_utils::runGlobalInitialization(std::move(db), &workspace);
            const auto net_init = caffe2::predictor_utils::getNet(
                *meta_net.get(),
                caffe2::PredictorConsts::default_instance().predict_init_net_type());

            CAFFE_ENFORCE(workspace.RunNetOnce(net_init));

            auto net = caffe2::NetDef(caffe2::predictor_utils::getNet(
                *meta_net.get(),
                caffe2::PredictorConsts::default_instance().predict_net_type()));

            auto res = workspace.CreateNet(net);
            std::cout << "Test" << res->Name() << std::endl;

            CAFFE_ENFORCE(workspace.CreateNet(net));

            net_graph = net;
            initialized = true;
        }

        return 0;
    }

    void predict(std::map<std::string, std::vector<double>> &result, const std::string &doc) {

        // Pre-process: tokenize input doc
        std::vector<string> tokens;
        std::string docCopy = doc;
        tokenize(tokens, docCopy);

        // Feed input to model as tensors
        caffe2::Tensor tensor_val = caffe2::TensorCPUFromValues<string>(
            {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens});
        BlobGetMutableTensor(workspace.CreateBlob("tokens_vals_str:value"), caffe2::CPU)
            ->CopyFrom(tensor_val);
        caffe2::Tensor tensor_lens = caffe2::TensorCPUFromValues<int>(
            {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())});
        BlobGetMutableTensor(workspace.CreateBlob("tokens_lens"), caffe2::CPU)
            ->CopyFrom(tensor_lens);

        std::printf("1\n");
        std::cout << net_graph.IsInitialized() << std::endl;

        // Run the model
        auto res = workspace.RunNet(net_graph.name());

        std::printf("1.5\n");

        CAFFE_ENFORCE(res);

        std::printf("2\n");

        // Extract and populate results into the response
        for (int i = 0; i < net_graph.external_output().size(); i++) {
            std::string label = net_graph.external_output()[i];
            result[label] = std::vector<double>();
            caffe2::Tensor tensor_scores = workspace.GetBlob(label)->Get<caffe2::Tensor>();
            for (int j = 0; j < tensor_scores.numel(); j++) {
                double score = tensor_scores.data<double>()[j];
                result[label].push_back(score);
            }
        }
    }

    void print_prediction(std::map<std::string, std::vector<double>> &labelScores) {
        std::cout << "Printing..." << std::endl;
        std::stringstream out;
        std::map<std::string, std::vector<double>>::iterator it;
        for (it = labelScores.begin(); it != labelScores.end(); it++) {
            out << it->first << ":";
            for (int i = 0; i < it->second.size(); i++) {
                out << it->second.at(i) << " ";
            }
            out << std::endl;
        }
        std::cout << out.str();
    }

    int main(int argc, char **argv) {

        // Parse command line args
        if (argc < 2) {
            std::cerr << "Usage:" << std::endl;
            std::cerr << "./predictor-cli <miniDB file>" << std::endl;
            return 1;
        }

        std::string model_file = argv[1];
        std::map<std::string, std::vector<double>> label_scores;
        
        if(!load_model(model_file.c_str())) {
            std::printf("Cannot open %s\n", model_file.c_str());
            return -1;
        }
        
        std::string input;
        while (std::getline(std::cin, input)) {
            predict(label_scores, input);
            std::printf("Predicted\n");
            print_prediction(label_scores);
        }

        return 0;
    }
}

// // Main handler for the predictor cli
// class PredictorHandlerCli {
//   private:
//     NetDef mPredictNet;
//     Workspace mWorkspace;

//     NetDef loadAndInitModel(Workspace& workspace, string& modelFile) {
//       auto db = unique_ptr<DBReader>(new DBReader("minidb", modelFile));
//       auto metaNetDef = runGlobalInitialization(move(db), &workspace);
//       const auto predictInitNet = getNet(
//         *metaNetDef.get(),
//         PredictorConsts::default_instance().predict_init_net_type()
//       );
//       CAFFE_ENFORCE(workspace.RunNetOnce(predictInitNet));

//       auto predictNet = NetDef(getNet(
//         *metaNetDef.get(),
//         PredictorConsts::default_instance().predict_net_type()
//       ));
//       CAFFE_ENFORCE(workspace.CreateNet(predictNet));

//       return predictNet;
//     }

//     void tokenize(vector<string>& tokens, string& doc) {
//       transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
//       int start = 0;
//       int end = 0;
//       for (int i = 0; i < doc.length(); i++) {
//         if (isspace(doc.at(i))){
//           end = i;
//           if (end != start) {
//             tokens.push_back(doc.substr(start, end - start));
//           }

//           start = i + 1;
//         }
//       }

//       if (start < doc.length()) {
//         tokens.push_back(doc.substr(start, doc.length() - start));
//       }

//       if (tokens.size() == 0) {
//         // Add PAD_TOKEN in case of empty text
//         tokens.push_back("<pad>");
//       }
//     }

//   public:
//     PredictorHandlerCli(string &modelFile): mWorkspace("workspace") {
//       mPredictNet = loadAndInitModel(mWorkspace, modelFile);
//     }

//     void predict(map<string, vector<double>>& _return, const string& doc) {
//       // Pre-process: tokenize input doc
//       vector<string> tokens;
//       string docCopy = doc;
//       tokenize(tokens, docCopy);

//       // Feed input to model as tensors
//       Tensor valTensor = TensorCPUFromValues<string>(
//         {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens}
//       );
//       BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_vals_str:value"), CPU)
//         ->CopyFrom(valTensor);
//       Tensor lensTensor = TensorCPUFromValues<int>(
//         {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())}
//       );
//       BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_lens"), CPU)
//         ->CopyFrom(lensTensor);

//       // Run the model
//       CAFFE_ENFORCE(mWorkspace.RunNet(mPredictNet.name()));

//       // Extract and populate results into the response
//       for (int i = 0; i < mPredictNet.external_output().size(); i++) {
//         string label = mPredictNet.external_output()[i];
//         _return[label] = vector<double>();
//         Tensor scoresTensor = mWorkspace.GetBlob(label)->Get<Tensor>();
//         for (int j = 0; j < scoresTensor.numel(); j++) {
//           float score = scoresTensor.data<float>()[j];
//           _return[label].push_back(score);
//         }
//       }
//     }

//     void printPrediction(map<string, vector<double>>& labelScores){
//       stringstream out;
//       map<string, vector<double>>::iterator it;
//       for (it = labelScores.begin(); it != labelScores.end(); it++) {
//         out << it->first << ":";
//         for (int i = 0; i < it->second.size(); i++) {
//           out << it->second.at(i) << " ";
//         }

//         out << endl;
//       }
//       cout << out.str();
//     }
// };

// int main(int argc, char **argv) {
//   // Parse command line args
//   if (argc < 2) {
//     cerr << "Usage:" << endl;
//     cerr << "./predictor-cli <miniDB file>" << endl;
//     return 1;
//   }

//   string modelFile = argv[1];  
//   map<string, vector<double>> labelScores;
//   shared_ptr<PredictorHandlerCli> model(new PredictorHandlerCli(modelFile));
//   string input;
//   while (getline(cin,input)) {
//     model->predict(labelScores,input);
//     model->printPrediction(labelScores);
//   }

//   return 0;
// }
