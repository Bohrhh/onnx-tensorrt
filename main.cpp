/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "NvOnnxParser.h"
#include "onnx_utils.hpp"
#include "common.hpp"
#include "backend.hpp"

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

void print_usage() {
  cout << "ONNX to TensorRT model parser" << endl;
  cout << "Usage: onnx2trt onnx_model.onnx" << "\n"
       << "                [-o engine_file.trt]  (output TensorRT engine)" << "\n"
       << "                [-b max_batch_size (default 32)]" << "\n"
       << "                [-w max_workspace_size_bytes (default 1 GiB)]" << "\n"
       << "                [-d model_data_type_bit_depth] (32 => float32, 16 => float16)" << "\n"
       << "                [-D dynamic_range_file] (file for setting dynamic range)" << "\n"
       << "                [-l] (list network tensor names)" << "\n"
       << "                [-g] (debug mode)" << "\n"
       << "                [-v] (increase verbosity)" << "\n"
       << "                [-q] (decrease verbosity)" << "\n"
       << "                [-V] (show version information)" << "\n"
       << "                [-h] (show help)" << endl;
}


int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  TRTBackendParams Tparams;

  int arg = 0;
  while( (arg = ::getopt(argc, argv, "o:b:w:d:l:D:gvqVh")) != -1 ) {
    switch (arg){
    case 'o':
      if( optarg ) { Tparams.engine_filename = optarg; break; }
      else { cerr << "------ERROR: -o flag requires argument" << endl; return -1; }
    case 'b':
      if( optarg ) { Tparams.max_batch_size = atoll(optarg); break; }
      else { cerr << "------ERROR: -b flag requires argument" << endl; return -1; }
    case 'w':
      if( optarg ) { Tparams.max_workspace_size = atoll(optarg); break; }
      else { cerr << "------ERROR: -w flag requires argument" << endl; return -1; }
    case 'd':
      if( optarg ) { Tparams.model_dtype_nbits = atoi(optarg); break; }
      else { cerr << "------ERROR: -d flag requires argument" << endl; return -1; }
    case 'l':
      if( optarg ) { Tparams.layer_info = optarg; break; }
      else { cerr << "------ERROR: -l flag requires argument" << endl; return -1; }
    case 'D':
      if( optarg ) { Tparams.dynamic_range_file = optarg; break; }
      else { cerr << "------ERROR: -D flag requires argument" << endl; return -1; }
    case 'g': Tparams.debug_builder = true; break;
    case 'v': ++Tparams.verbosity; break;
    case 'q': --Tparams.verbosity; break;
    case 'V': common::print_version(); return 0;
    case 'h': print_usage(); return 0;
    }
  }
  int num_args = argc - optind;
  if( num_args != 1 ) {
    print_usage();
    return 0;
  }

  Tparams.onnx_filename = argv[optind];

  TRTBackend TBackend(Tparams);



// -------------------------------------------------------------------------
// load onnx model
// -------------------------------------------------------------------------
  if (TBackend.loadOnnxModel())
  {
    cout << "------PASSED: load onnx successfully!" << endl;
    TBackend.onnxInfo();
  }else
  {
    cerr << "------ERROR: failed to load onnx!" << endl;
    return -1;
  }
  

// -------------------------------------------------------------------------
// parser onnx to tensorrt network
// -------------------------------------------------------------------------
  if (TBackend.parserOnnx())
  {
    cout << "------PASSED: parser onnx to tensorrt network successfully!" << endl;
  }else
  {
    cerr << "------ERROR: failed to parser onnx to tensorrt network!" << endl;
    return -1;
  }


// -------------------------------------------------------------------------
// export network tensor names
// -------------------------------------------------------------------------
  if (!Tparams.layer_info.empty())
  {
    if(TBackend.layerInfo()){
      cout << "------PASSED: generate network tensor names successfully! Writing: " << Tparams.layer_info << endl;
      return 0;
    }
    else{
      cerr << "------ERROR: failed to export network tensor names!" << endl;
      return -1;
    }
  }


// -------------------------------------------------------------------------
// using builder to optimize network and generate engine
// -------------------------------------------------------------------------
  if (!Tparams.engine_filename.empty())
  {
    if(TBackend.build()){
      cout << "------PASSED: generate the engine successfully!" << endl;
      return 0;
    }
    else{
      cerr << "------ERROR: failed to build the engine!" << endl;
      return -1;
    }
  }

  
  return 0;
}
