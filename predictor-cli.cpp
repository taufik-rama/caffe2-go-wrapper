// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <map>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/predictor/predictor_utils.h"
#include "caffe2/proto/predictor_consts.pb.h"
#include "caffe2/utils/proto_utils.h"

using namespace std;
using namespace caffe2;
using namespace db;
using namespace predictor_utils;

// Main handler for the predictor cli
class PredictorHandlerCli {
  private:
    NetDef mPredictNet;
    Workspace mWorkspace;

    NetDef loadAndInitModel(Workspace& workspace, string& modelFile) {
      auto db = unique_ptr<DBReader>(new DBReader("minidb", modelFile));
      auto metaNetDef = runGlobalInitialization(move(db), &workspace);
      const auto predictInitNet = getNet(
        *metaNetDef.get(),
        PredictorConsts::default_instance().predict_init_net_type()
      );
      CAFFE_ENFORCE(workspace.RunNetOnce(predictInitNet));

      auto predictNet = NetDef(getNet(
        *metaNetDef.get(),
        PredictorConsts::default_instance().predict_net_type()
      ));
      CAFFE_ENFORCE(workspace.CreateNet(predictNet));

      return predictNet;
    }

    void tokenize(vector<string>& tokens, string& doc) {
      transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
      int start = 0;
      int end = 0;
      for (int i = 0; i < doc.length(); i++) {
        if (isspace(doc.at(i))){
          end = i;
          if (end != start) {
            tokens.push_back(doc.substr(start, end - start));
          }

          start = i + 1;
        }
      }

      if (start < doc.length()) {
        tokens.push_back(doc.substr(start, doc.length() - start));
      }

      if (tokens.size() == 0) {
        // Add PAD_TOKEN in case of empty text
        tokens.push_back("<pad>");
      }
    }

  public:
    PredictorHandlerCli(string &modelFile): mWorkspace("workspace") {
      mPredictNet = loadAndInitModel(mWorkspace, modelFile);
    }

    void predict(map<string, vector<double>>& _return, const string& doc) {
      // Pre-process: tokenize input doc
      vector<string> tokens;
      string docCopy = doc;
      tokenize(tokens, docCopy);

      // Feed input to model as tensors
      Tensor valTensor = TensorCPUFromValues<string>(
        {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens}
      );
      BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_vals_str:value"), CPU)
        ->CopyFrom(valTensor);
      Tensor lensTensor = TensorCPUFromValues<int>(
        {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())}
      );
      BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_lens"), CPU)
        ->CopyFrom(lensTensor);

      // Run the model
      CAFFE_ENFORCE(mWorkspace.RunNet(mPredictNet.name()));

      // Extract and populate results into the response
      for (int i = 0; i < mPredictNet.external_output().size(); i++) {
        string label = mPredictNet.external_output()[i];
        _return[label] = vector<double>();
        Tensor scoresTensor = mWorkspace.GetBlob(label)->Get<Tensor>();
        for (int j = 0; j < scoresTensor.numel(); j++) {
          float score = scoresTensor.data<float>()[j];
          _return[label].push_back(score);
        }
      }
    }

    void printPrediction(map<string, vector<double>>& labelScores){
      stringstream out;
      map<string, vector<double>>::iterator it;
      for (it = labelScores.begin(); it != labelScores.end(); it++) {
        out << it->first << ":";
        for (int i = 0; i < it->second.size(); i++) {
          out << it->second.at(i) << " ";
        }

        out << endl;
      }
      cout << out.str();
    }
};

int main(int argc, char **argv) {
  // Parse command line args
  if (argc < 2) {
    cerr << "Usage:" << endl;
    cerr << "./predictor-cli <miniDB file>" << endl;
    return 1;
  }

  string modelFile = argv[1];  
  map<string, vector<double>> labelScores;
  shared_ptr<PredictorHandlerCli> model(new PredictorHandlerCli(modelFile));
  string input;
  while (getline(cin,input)) {
    model->predict(labelScores,input);
    model->printPrediction(labelScores);
  }

  return 0;
}
