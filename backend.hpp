#include "NvOnnxParser.h"
#include "onnx_utils.hpp"
#include "common.hpp"
#include "half.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <unistd.h> // For ::getopt
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <ctime>
#include <fcntl.h> // For ::open
#include <limits>
#include <unordered_map>


struct TRTBackendParams
{
  std::string onnx_filename;
  std::string engine_filename;
  std::string layer_info;
  std::string dynamic_range_file;
  std::string input_figures;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
  int model_dtype_nbits = 32;
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  bool debug_builder = false;
};


class TRTBackend
{

public:

  TRTBackend(const TRTBackendParams& params): Tparams(params) {}

  // -------------------------------------------------------------------------
  // load onnx model
  // -------------------------------------------------------------------------
  bool loadOnnxModel();


  // -------------------------------------------------------------------------
  // parser onnx to tensorrt network
  // -------------------------------------------------------------------------
  bool parserOnnx();


  // -------------------------------------------------------------------------
  // Write network tensors' names to a file, 
  // can be used to determine dynamic ranges for per tensor when quant
  // -------------------------------------------------------------------------
  bool layerInfo();


  // -------------------------------------------------------------------------
  // required by quantization
  // -------------------------------------------------------------------------
  void setLayerPrecision();
  bool setDynamicRange();
  

  // -------------------------------------------------------------------------
  // using builder to optimize network and generate engine
  // -------------------------------------------------------------------------
  bool build();


  // -------------------------------------------------------------------------
  // deserializing an engine and doing inference
  // -------------------------------------------------------------------------
  bool inference();


  // -------------------------------------------------------------------------
  // others
  // -------------------------------------------------------------------------
  bool readPerTensorDynamicRangeValues();
  void onnxInfo();


  // -------------------------------------------------------------------------
  // to do
  // bool inference(engine_file);
  // -------------------------------------------------------------------------

  


  TRTBackendParams Tparams;

private:

  ::ONNX_NAMESPACE::ModelProto onnx_model;
  common::TRT_Logger trt_logger;
  std::shared_ptr<nvinfer1::IBuilder> trt_builder = common::infer_object(nvinfer1::createInferBuilder(trt_logger));
  std::shared_ptr<nvinfer1::INetworkDefinition> trt_network = common::infer_object(trt_builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  std::shared_ptr<nvonnxparser::IParser> trt_parser  = common::infer_object(nvonnxparser::createParser(*trt_network, trt_logger));
  
  //!< Mapping from tensor name to max absolute dynamic range values
  std::unordered_map<std::string, float>mPerTensorDynamicRangeMap; 

  
};


bool TRTBackend::loadOnnxModel(){
  if (!std::ifstream(Tparams.onnx_filename.c_str())) {
    cerr << "------ERROR: input file not found, " << Tparams.onnx_filename << endl;
    return false;
  }

  bool is_binary = common::ParseFromFile_WAR(&onnx_model, Tparams.onnx_filename.c_str());
  if( !is_binary && !common::ParseFromTextFile(&onnx_model, Tparams.onnx_filename.c_str()) ) {
    cerr << "------ERROR: failed to parse ONNX model" << endl;
    return false;
  }
  return true;
}


void TRTBackend::onnxInfo(){

  int64_t opset_version = (onnx_model.opset_import().size() ?
                            onnx_model.opset_import(0).version() : 0);
  cout << "----------------------------------------------------------------" << endl;
  cout << "Input filename:   " << Tparams.onnx_filename << endl;
  cout << "ONNX IR version:  " << common::onnx_ir_version_string(onnx_model.ir_version()) << endl;
  cout << "Opset version:    " << opset_version << endl;
  cout << "Producer name:    " << onnx_model.producer_name() << endl;
  cout << "Producer version: " << onnx_model.producer_version() << endl;
  cout << "Domain:           " << onnx_model.domain() << endl;
  cout << "Model version:    " << onnx_model.model_version() << endl;
  cout << "Doc string:       " << onnx_model.doc_string() << endl;
  cout << "----------------------------------------------------------------" << endl;
          
}


bool TRTBackend::parserOnnx(){

  if( Tparams.verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "---------------------- Parsing Onnx Model ----------------------" << endl;
  }


  std::ifstream onnx_file(Tparams.onnx_filename.c_str(),
                          std::ios::binary | std::ios::ate);
  std::streamsize file_size = onnx_file.tellg();
  onnx_file.seekg(0, std::ios::beg);
  std::vector<char> onnx_buf(file_size);
  if( !onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) {
    cerr << "------ERROR: Failed to read from file " << Tparams.onnx_filename << endl;
    return false;
  }
  if( !trt_parser->parse(onnx_buf.data(), onnx_buf.size()) ) {
    int nerror = trt_parser->getNbErrors();
    for( int i=0; i<nerror; ++i ) {
      nvonnxparser::IParserError const* error = trt_parser->getError(i);
      if( error->node() != -1 ) {
        ::ONNX_NAMESPACE::NodeProto const& node =
          onnx_model.graph().node(error->node());
        cerr << "While parsing node number " << error->node()
              << " [" << node.op_type();
        if( node.output().size() ) {
          cerr << " -> \"" << node.output(0) << "\"";
        }
        cerr << "]:" << endl;
        if( Tparams.verbosity >= (int)nvinfer1::ILogger::Severity::kINFO ) {
          cerr << "--- Begin node ---" << endl;
          cerr << node << endl;
          cerr << "--- End node ---" << endl;
        }
      }
      cerr << "------ERROR: "
            << error->file() << ":" << error->line()
            << " In function " << error->func() << ":\n"
            << "[" << static_cast<int>(error->code()) << "] " << error->desc()
            << endl;
    }
    return false;
  }
  return true;

}


bool TRTBackend::layerInfo(){
  if( !Tparams.layer_info.empty() ){
    cout << "In order to run Int8 inference without calibration, "
            "user will need to provide dynamic range for all the network tensors."
            << endl;

    std::ofstream tensorsFile{Tparams.layer_info};

    // Iterate through network inputs to write names of input tensors.
    for (int i = 0; i < trt_network->getNbInputs(); ++i)
    {
        std::string tName = trt_network->getInput(i)->getName();
        tensorsFile << "TensorName: " << tName << endl;
    }

    // Iterate through network layers.
    for (int i = 0; i < trt_network->getNbLayers(); ++i)
    {
        // Write output tensors of a layer to the file.
        for (int j = 0; j < trt_network->getLayer(i)->getNbOutputs(); ++j)
        {
            std::string tName = trt_network->getLayer(i)->getOutput(j)->getName();
            tensorsFile << "TensorName: " << tName << endl;
        }
    }
    tensorsFile.close();

    return true;
  }

  return false;

}


bool TRTBackend::readPerTensorDynamicRangeValues()
{
    std::ifstream iDynamicRangeStream(Tparams.dynamic_range_file);
    if (!iDynamicRangeStream)
    {
        cout << "Could not find per tensor scales file: " << Tparams.dynamic_range_file << endl;
        return false;
    }

    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line))
    {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        float dynamicRange = std::stof(token);
        mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
    }
    return true;
}



bool TRTBackend::setDynamicRange()
{
    // populate per tensor dynamic range
    if (!readPerTensorDynamicRangeValues())
    {
        return false;
    }

    if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
    {
        cout << "If dynamic range for a tensor is missing, TensorRT will run inference assuming dynamic range for "
                    "the tensor as optional."
                 << endl;
        cout << "If dynamic range for a tensor is required then inference will fail. Please generate "
                    "missing per tensor dynamic range."
                 << endl;
    }
    // set dynamic range for network input tensors
    for (int i = 0; i < trt_network->getNbInputs(); ++i)
    {
        std::string tName = trt_network->getInput(i)->getName();
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
        {
            trt_network->getInput(i)->setDynamicRange(-mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
        }
        else
        {
            if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
            {
                cout << "------Warning: " << "Missing dynamic range for tensor: " << tName << endl;
            }
        }
    }

     // set dynamic range for layer output tensors
    for (int i = 0; i < trt_network->getNbLayers(); ++i)
    {
        auto lyr = trt_network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            std::string tName = lyr->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                lyr->getOutput(j)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
            }
            else if (lyr->getType() == nvinfer1::LayerType::kCONSTANT)
            {
                nvinfer1::IConstantLayer* cLyr = static_cast<nvinfer1::IConstantLayer*>(lyr);
                if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
                {
                    cout << "------Warning: " << "Computing missing dynamic range for tensor, " << tName << ", from weights."
                                << endl;
                }
                auto wts = cLyr->getWeights();
                double max = std::numeric_limits<double>::min();
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb)
                {
                    double val;
                    switch (wts.type)
                    {
                    case nvinfer1::DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    }
                    max = std::max(max, std::abs(val));
                }

                lyr->getOutput(j)->setDynamicRange(-max, max);
            }
            else
            {
                if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
                {
                    cout << "------Warning: " << "Missing dynamic range for tensor: " << tName << endl;
                }
            }
        }
    }

    // set dynamic range for layer output tensors
    for (int i = 0; i < trt_network->getNbLayers(); ++i)
    {
        for (int j = 0; j < trt_network->getLayer(i)->getNbOutputs(); ++j)
        {
            std::string tName = trt_network->getLayer(i)->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                trt_network->getLayer(i)->getOutput(j)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
            }
            else
            {
                if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
                {
                    cout << "------Warning: " << "Missing dynamic range for tensor: " << tName << endl;
                }
            }
        }
    }

    if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
    {
        cout << "Per Tensor Dynamic Range Values for the Network:" << endl;
        for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
            cout << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second << endl;
    }
    return true;
}



void TRTBackend::setLayerPrecision()
{
    for (int i = 0; i < trt_network->getNbLayers(); ++i)
    {
        auto layer = trt_network->getLayer(i);
        if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
        {
            std::string layerName = layer->getName();
            cout << "Layer: " << layerName << ". Precision: INT8" << endl;
        }

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        if (layer->getType() != nvinfer1::LayerType::kCONSTANT
                && layer->getType() != nvinfer1::LayerType::kCONCATENATION
                && layer->getType() != nvinfer1::LayerType::kSHAPE
                && layer->getType() != nvinfer1::LayerType::kSLICE
                && layer->getType() != nvinfer1::LayerType::kGATHER
                && layer->getType() != nvinfer1::LayerType::kSHUFFLE
                && layer->getType() != nvinfer1::LayerType::kIDENTITY
                && layer->getType() != nvinfer1::LayerType::kPLUGIN
                && layer->getType() != nvinfer1::LayerType::kPLUGIN_V2)
        {
            // set computation precision of the layer
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            if (Tparams.verbosity>(int)nvinfer1::ILogger::Severity::kWARNING)
            {
                std::string tensorName = layer->getOutput(j)->getName();
                cout << "Tensor: " << tensorName << ". OutputType: INT8" << endl;
            }
            // set output type of execution tensors and not shape tensors.
            if (layer->getOutput(j)->isExecutionTensor())
            {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}


bool TRTBackend::build()
{

  nvinfer1::DataType model_dtype;
  if(      Tparams.model_dtype_nbits == 32 ) { model_dtype = nvinfer1::DataType::kFLOAT; }
  else if( Tparams.model_dtype_nbits == 16 ) { model_dtype = nvinfer1::DataType::kHALF; }
  else if( Tparams.model_dtype_nbits ==  8 ) { model_dtype = nvinfer1::DataType::kINT8; }
  else {
    cerr << "------ERROR: invalid model data type bit depth, " << Tparams.model_dtype_nbits << endl;
    return false;
  }

  auto config = common::infer_object(trt_builder->createBuilderConfig());
  if (!config)
  {
      cerr << "------ERROR: unable to create config object." << endl;
      return false;
  }


  bool fp16 = trt_builder->platformHasFastFp16();
  // Configure buider
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  config->setMaxWorkspaceSize(Tparams.max_workspace_size);

  if( fp16 && model_dtype == nvinfer1::DataType::kHALF)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  else if (model_dtype == nvinfer1::DataType::kINT8 && !Tparams.dynamic_range_file.empty()){
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(nullptr);
  }


  if( Tparams.verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "Building TensorRT engine, FP16 available:"<< fp16 << endl;
    cout << "    Max batch size:     " << Tparams.max_batch_size << endl;
    cout << "    Max workspace size: " << Tparams.max_workspace_size / (1024. * 1024) << " MiB" << endl;
  }
  trt_builder->setMaxBatchSize(Tparams.max_batch_size);



  // int8 prepare
  if (Tparams.model_dtype_nbits==8 && !Tparams.dynamic_range_file.empty()){
    // force layer to execute with required precision
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    setLayerPrecision();

    if (setDynamicRange()){
      cout << "------PASSED: set dynamic range successfully!" << endl;
    }
    else{
      cerr << "------ERROR: failed to set dynamic range!" << endl;
      return false;
    }
  }
  else if(Tparams.model_dtype_nbits==8 && Tparams.dynamic_range_file.empty()){
    cerr << "------ERROR: should provide dynamic range file when using int8 mode!" << endl;
    return false;
  }


  // build TRT engine
  trt_builder->setDebugSync(Tparams.debug_builder);
  // auto trt_engine = common::infer_object(trt_builder->buildCudaEngine(*trt_network.get()));
  auto trt_engine = common::infer_object(trt_builder->buildEngineWithConfig(*trt_network, *config));
  if (!trt_engine)
  {
      cerr << "------ERROR: unable to build cuda engine." << endl;
      return false;
  }

  // inference
  


  // serialize the engine
  auto engine_plan = common::infer_object(trt_engine->serialize());
  std::ofstream engine_file(Tparams.engine_filename.c_str());
  if (!engine_file) {
    cerr << "------ERROR: failed to open output file for writing: "
          << Tparams.engine_filename << endl;
    return false;
  }
  if( Tparams.verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "Writing TensorRT engine to " << Tparams.engine_filename << endl;
  }
  engine_file.write((char*)engine_plan->data(), engine_plan->size());
  engine_file.close();
  return true;
  
  
}