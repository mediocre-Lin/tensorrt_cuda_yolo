#include "trt_infer.h"
Logger pxNvLogger = Logger();

bool BaseInfer::SetUp(const std::string& model_path, const int work_space,
                      const bool fp16, const int batch_num, const int det_width,
                      const int det_height, const std::vector<float> means,
                      const std::vector<float> stds) {
  return false;
}

bool BaseInfer::Infer(const std::vector<cv::Mat>& mat) { return false; }

bool BaseInfer::GetRes() { return false; }

bool BaseInfer::ParseONNX(const std::string& model_path) {
  builder_ = nvinfer1::createInferBuilder(pxNvLogger);
  network_ = builder_->createNetworkV2(1U);
  parser_ = nvonnxparser::createParser(*network_, pxNvLogger);
  parser_->parseFromFile(model_path.c_str(), 2);
  return true;
}

bool BaseInfer::BuildEngine(const int workspace_size, bool fp16) {
  config_ = builder_->createBuilderConfig();
  if (fp16 && !builder_->platformHasFastFp16()) {
    std::cout << "WARN: your platform has no fastFP16, will try to turn to "
                 "TF32 Mode"
              << std::endl;
    fp16 = false;
  }
  if (fp16) {
    std::cout << "Build with FP16 Mode" << std::endl;
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    // config_->setPreviewFeature(nvinfer1::PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, 0);
    // config_.back()->setAvgTimingIterations(16);
    config_->setAvgTimingIterations(16);
  } else {
    if (!builder_->platformHasTf32()) {
      std::cout << "WARN: your platform has no TF32, will turn to Normal Mode"
                << std::endl;
    } else {
      std::cout << "Build with TF32 Mode" << std::endl;
      config_->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
  }
  config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                              1U << workspace_size);
  engine_ = builder_->buildEngineWithConfig(*network_, *config_);
  return true;
}

bool BaseInfer::SaveTrtSerialized(const std::string& save2path) {
  const char* kPtext = strrchr(save2path.c_str(), '.');
  std::string postfix_text(kPtext);
  serialize_model_ = engine_->serialize();
  FILE* out_file;
  std::string save_path = save2path;
  fopen_s(&out_file,
          (save_path.replace(save_path.size() - postfix_text.size(),
                             postfix_text.size(), ".trt"))
              .c_str(),
          "wb");

  fwrite(serialize_model_->data(), 1, serialize_model_->size(), out_file);
  fclose(out_file);
  delete config_;
  delete builder_;
  delete network_;
  delete parser_;
  delete serialize_model_;
  return true;
}

bool BaseInfer::LoadTrtFile(const std::string& model_path) {
  std::ifstream ifile(model_path, std::ios::in | std::ios::binary);
  if (!ifile) {
    std::cout << "read serialzed file failed : " << model_path << std::endl;
    return false;
  }
  ifile.seekg(0, std::ios::end);
  const int kMdsize = ifile.tellg();
  ifile.clear();
  ifile.seekg(0, std::ios::beg);
  std::vector<char> buf(kMdsize);
  ifile.read(&buf[0], kMdsize);
  ifile.close();
  runtime_ = nvinfer1::createInferRuntime(pxNvLogger);
  initLibNvInferPlugins(&pxNvLogger, "");
  engine_ = runtime_->deserializeCudaEngine((void*)&buf[0], kMdsize, nullptr);
  return true;
}

bool BaseInfer::Infer(const cv::cuda::GpuMat& d_mat, const std::vector<cv::Rect>& rois)
{
    return false;
}
