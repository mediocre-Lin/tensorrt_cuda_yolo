#ifndef TRT_INFER_H
#define TRT_INFER_H
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/opencv.hpp"
#define TIMERSTART(tag) \
  auto tag##_start = std::chrono::steady_clock::now(), tag##_end = tag##_start
#define TIMEREND(tag) tag##_end = std::chrono::steady_clock::now()
#define DURATION_s(tag)                                                \
  printf("%s costs %d s\n", #tag,                                      \
         std::chrono::duration_cast<std::chrono::seconds>(tag##_end -  \
                                                          tag##_start) \
             .count())
#define DURATION_ms(tag)                                                    \
  printf("%s costs %d ms\n", #tag,                                          \
         std::chrono::duration_cast<std::chrono::milliseconds>(tag##_end -  \
                                                               tag##_start) \
             .count());
#define DURATION_us(tag)                                                    \
  printf("%s costs %d us\n", #tag,                                          \
         std::chrono::duration_cast<std::chrono::microseconds>(tag##_end -  \
                                                               tag##_start) \
             .count());
#define DURATION_ns(tag)                                                   \
  printf("%s costs %d ns\n", #tag,                                         \
         std::chrono::duration_cast<std::chrono::nanoseconds>(tag##_end -  \
                                                              tag##_start) \
             .count());

struct DetInstance {
  int id;
  float score;
  int bbox_ltx;
  int bbox_lty;
  int bbox_rbx;
  int bbox_rby;
};

struct DetPredictorOutput {
  std::vector<DetInstance> instances;
};

#define CUDACHECK(call)                                                    \
  {                                                                        \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess) {                                            \
      printf("Error: %s : Line: %d \n", __FILE__, __LINE__);               \
      printf("code: %d ,reason : %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                             \
    }                                                                      \
  }

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    if (severity != nvinfer1::ILogger::Severity::kINFO &&
        severity != nvinfer1::ILogger::Severity::kVERBOSE) {
      std::cout << msg << std::endl;
    }
  };
};
extern Logger pxNvLogger;

class BaseInfer {
 public:
  virtual bool SetUp(const std::string& model_path, const int work_space,
                     const bool fp16, const int batch_num, const int det_width,
                     const int det_height, const std::vector<float> means,
                     const std::vector<float> stds);
  virtual bool Infer(const std::vector<cv::Mat>& mat);
  virtual bool Infer(const cv::cuda::GpuMat& d_mat, const std::vector<cv::Rect>& rois);
  virtual bool GetRes();
  virtual bool ParseONNX(const std::string& model_path);
  virtual bool BuildEngine(const int workspace_size, bool fp16 = true);
  virtual bool SaveTrtSerialized(const std::string& save2path);
  virtual bool LoadTrtFile(const std::string& model_path);
  virtual ~BaseInfer(){};

 protected:
  nvinfer1::IRuntime* runtime_;
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;
  nvonnxparser::IParser* parser_;
  nvinfer1::IBuilderConfig* config_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IHostMemory* serialize_model_;
};
#endif
