#include "yolov8.h"

extern "C" void toNCHW_Norm(const cv::cuda::GpuMat &input, float *output,
                            float *means, float *stds, cudaStream_t stream);
extern "C" void ROI_BGR2RGB_NORM(const cv::cuda::GpuMat &input, float *output,
                                 const float *means, const float *stds,
                                 int roi_x, int roi_y, int roi_width,
                                 int roi_height, cudaStream_t stream);
extern "C" void Batch_ROI_BGR2RGB_NORM(const cv::cuda::GpuMat &input,
                                       float *output, int batch, int roi_width,
                                       int roi_height, float *d_means,
                                       float *d_stds, int *d_x, int *d_y,
                                       int *d_w, int *d_h,
                                       const cudaStream_t stream);
bool YOLOv8Infer::SetUp(const std::string &model_path, const int work_space,
                        const bool fp16, const int batch_num,
                        const int det_width, const int det_height,
                        const std::vector<float> means,
                        const std::vector<float> stds) {
  det_width_ = det_width;
  det_height_ = det_height;
  batch_num_ = batch_num;
  means_ = means;
  stds_ = stds;
  cudaMalloc(&d_means, 3 * sizeof(float));
  cudaMalloc(&d_stds, 3 * sizeof(float));
  h_x = new int[batch_num];
  h_y = new int[batch_num];
  h_w = new int[batch_num];
  h_h = new int[batch_num];
  cudaMallocHost((void**)&h_x,batch_num * sizeof(int));
  cudaMallocHost((void**)&h_y,batch_num * sizeof(int));
  cudaMallocHost((void**)&h_w,batch_num * sizeof(int));
  cudaMallocHost((void**)&h_h,batch_num * sizeof(int));

  cudaMalloc(&d_x, batch_num * sizeof(int));
  cudaMalloc(&d_y, batch_num * sizeof(int));
  cudaMalloc(&d_w, batch_num * sizeof(int));
  cudaMalloc(&d_h, batch_num * sizeof(int));
  // streams_mat_.resize(batch_num);
  d_mats_.resize(batch_num);
  CUDACHECK(cudaStreamCreate(&stream_));
  cudaMemcpyAsync(d_means, means.data(), 3 * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_stds, stds.data(), 3 * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // cuda_streams_.resize(batch_num);
  big_d_mat = cv::cuda::GpuMat(8000, 51200, CV_8UC3);
  stream_mat_ = cv::cuda::StreamAccessor::wrapStream(stream_);
  // cudaMallocPitch((void **)&d_mat_data_, (size_t *)&pitch_,
  //                 (size_t)(50000) * sizeof(uchar3), (size_t)(8000));
  // for (int stream_i = 0; stream_i < batch_num; ++stream_i) {
  // streams_mat_[stream_i] = cv::cuda::Stream::Stream();
  // CUDACHECK(cudaStreamCreate(&cuda_streams_[stream_i]));
  // }
  const char *kPtext = strrchr(model_path.c_str(), '.');
  std::string c_text(kPtext);
  std::cout << model_path << std::endl;
  if (c_text == std::string(".onnx")) {
    ParseONNX(model_path);
    NMSPlugin();
    BuildEngine(work_space, fp16);
    SaveTrtSerialized(model_path);

  } else if (c_text == std::string(".trt")) {
    LoadTrtFile(model_path);
  }
  context_ = engine_->createExecutionContext();
  InitIO();
  return false;
}

bool YOLOv8Infer::Infer(const std::vector<cv::Mat> &mat) {
  ori_width_ = mat[0].cols;
  ori_height_ = mat[0].rows;
  return EffInfer(mat);
}

bool YOLOv8Infer::Infer(const cv::cuda::GpuMat &d_mat,
                        const std::vector<cv::Rect> &rois) {
  ori_width_ = rois[0].width;
  ori_height_ = rois[0].height;
  return EffInfer(d_mat, rois);
}

bool YOLOv8Infer::Infer(const std::vector<cv::Rect> &rois) {
  ori_width_ = rois[0].width;
  ori_height_ = rois[0].height;
  return EffInfer(rois);
}

void YOLOv8Infer::LoadData(const cv::Mat &data) {
  cudaMemcpy2DAsync(d_mat_data_, pitch_, data.data, sizeof(uchar3) * data.cols,
                    sizeof(uchar3) * data.cols, data.rows,
                    cudaMemcpyHostToDevice, stream_);
}

void YOLOv8Infer::LoadData(const cv::cuda::HostMem &data) {
  big_d_mat.upload(data, stream_mat_);
}

bool YOLOv8Infer::GetRes(std::vector<DetPredictorOutput> &res) {
  cudaStreamSynchronize(stream_);
  res.resize(batch_num_);
  // std::cout << "batch_num_ :" << batch_num_ << std::endl;

  for (int b_i = 0; b_i < batch_num_; ++b_i) {
    res[b_i].instances.clear();
    int x_offset = (det_width_ * scale_ - ori_width_) / 2;
    int y_offset = (det_height_ * scale_ - ori_height_) / 2;
    // std::cout << b_i << ":"
    //           << "num_dets :" << num_dets_[b_i] << std::endl;
    for (size_t i = 0; i < num_dets_[b_i]; i++) {
      DetInstance ins;
      float ins_score = det_scores_[b_i * 100 + i];
      // std::cout << i << ":"<< "ins_score :" << ins_score << std::endl;
      float x0 = (det_boxes_[i * 4 + b_i * 4 * 100]) * scale_ - x_offset;
      // std::cout << i << ":"
      //     << "x0 :" << x0 << std::endl;
      float y0 = (det_boxes_[i * 4 + 1 + b_i * 4 * 100]) * scale_ - y_offset;
      // std::cout << i << ":"
      //     << "y0 :" << y0 << std::endl;
      float x1 = (det_boxes_[i * 4 + 2 + b_i * 4 * 100]) * scale_ - x_offset;

      // std::cout << i << ":"
      //           << "x1 :" << x1 << std::endl;
      float y1 = (det_boxes_[i * 4 + 3 + b_i * 4 * 100]) * scale_ - y_offset;
      // std::cout << i << ":"
      //     << "y1 :" << y1 << std::endl;
      x0 = std::max(std::min(x0, (float)(ori_width_ - 1)), 0.f);
      y0 = std::max(std::min(y0, (float)(ori_height_ - 1)), 0.f);
      x1 = std::max(std::min(x1, (float)(ori_width_ - 1)), 0.f);
      y1 = std::max(std::min(y1, (float)(ori_height_ - 1)), 0.f);
      ins.score = ins_score;
      ins.id = det_classes_[b_i * 100 + i];
      ins.bbox_ltx = x0;
      ins.bbox_lty = y0;
      ins.bbox_rbx = x1;
      ins.bbox_rby = y1;
      // std::cout << i << ":"
      //     << "id :" << det_classes_[b_i * 100 + i] << std::endl;

      res[b_i].instances.push_back(ins);
      // std::cout << "push" << std::endl;
    }
  }

  return true;
}

cudaStream_t YOLOv8Infer::GetCudaStream() { return stream_; }

YOLOv8Infer::~YOLOv8Infer(){
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_w);
  cudaFreeHost(h_h);
  delete num_dets_;
  delete det_scores_;
  delete det_classes_;
  delete in_data_;
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_w);
  cudaFree(d_h);
  cudaFree(d_means);
  cudaFree(d_stds);
  cudaStreamDestroy(stream_);
  for (int i = 0; i < 5; ++i) {
    cudaFree(buffs_[i]);
  }
  delete context_;
}

bool YOLOv8Infer::NavieInfer(const std::vector<cv::Mat> &mat) { return false; }

bool YOLOv8Infer::EffInfer(const std::vector<cv::Mat> &mat) {
  // for (int i = 0; i < mat.size(); ++i) {
  //   d_mats_[i].upload(mat[i], streams_mat_[i]);
  //   D_Preprocess(d_mats_[i], streams_mat_[i]);
  //   toNCHW_Norm(d_mats_[i],
  //               (float *)buffs_[0] + i * 3 * det_height_ * det_width_,
  //               means_.data(), stds_.data(),
  //               cv::cuda::StreamAccessor::getStream(streams_mat_[i]));
  // }

  // for (int ei = 0; ei < mat.size(); ++ei) {
  //   streams_mat_[ei].waitForCompletion();
  // }

  // context_->enqueueV2(&buffs_[0], stream_, nullptr);
  // CUDACHECK(cudaMemcpyAsync(num_dets_, buffs_[1], out_size_[0] * sizeof(int),
  //                           cudaMemcpyDeviceToHost, stream_));
  // CUDACHECK(cudaMemcpyAsync(det_boxes_, buffs_[2], out_size_[1] * sizeof(float),
  //                           cudaMemcpyDeviceToHost, stream_));
  // CUDACHECK(cudaMemcpyAsync(det_scores_, buffs_[3],
  //                           out_size_[2] * sizeof(float),
  //                           cudaMemcpyDeviceToHost, stream_));
  // CUDACHECK(cudaMemcpyAsync(det_classes_, buffs_[4], out_size_[3] * sizeof(int),
  //                           cudaMemcpyDeviceToHost, stream_));

  return true;
}

bool YOLOv8Infer::EffInfer(const std::vector<cv::Rect> &rois) {
  for (int i = 0; i < rois.size(); ++i) {
    h_y[i] = rois[i].y;
    h_x[i] = rois[i].x;
    h_w[i] = rois[i].width;
    h_h[i] = rois[i].height;
  }
  cudaMemcpyAsync(d_x, h_x, rois.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(d_y, h_y, rois.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(d_w, h_w, rois.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(d_h, h_h, rois.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  Batch_ROI_BGR2RGB_NORM(big_d_mat, (float *)buffs_[0], batch_num_,
                         rois[0].width, rois[0].height, d_means, d_stds, d_x,
                         d_y, d_w, d_h, stream_);
  // Batch_ROI_BGR2RGB_NORM_2(d_mat_data_, (float *)buffs_[0], batch_num_,
  //                          means_.data(), stds_.data(), rois, stream_);
  context_->enqueueV2(&buffs_[0], stream_, nullptr);
  CUDACHECK(cudaMemcpyAsync(num_dets_, buffs_[1], out_size_[0] * sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_boxes_, buffs_[2], out_size_[1] * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_scores_, buffs_[3],
                            out_size_[2] * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_classes_, buffs_[4], out_size_[3] * sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  // cudaStreamSynchronize(stream_);
  // TIMEREND(EffInfer);
  // DURATION_ms(EffInfer);

  return true;
}

bool YOLOv8Infer::EffInfer(const cv::cuda::GpuMat &d_mat,
                           const std::vector<cv::Rect> &rois) {
  // TIMERSTART(EffInfer);
  // Batch_ROI_BGR2RGB_NORM(d_mat, (float *)buffs_[0], batch_num_,
  // means_.data(),
  //                        stds_.data(), rois, stream_);
  // #pragma omp parallel for
  // for (int i = 0; i < rois.size(); ++i) {
  //   ROI_BGR2RGB_NORM(d_mat,(float *)buffs_[0] + i * 3 * det_height_ *
  //   det_width_, means_.data(), stds_.data(),
  //   rois[i].x,rois[i].y,rois[i].width,rois[i].height,cuda_streams_[i]);
  //   // d_mats_[i] =  d_mat(rois[i]);
  //   // D_Preprocess(d_mats_[i], streams_mat_[i]);
  //   // toNCHW_Norm(d_mats_[i],
  //   //             (float *)buffs_[0] + i * 3 * det_height_ * det_width_,
  //   //             means_.data(), stds_.data(),
  //   //             cv::cuda::StreamAccessor::getStream(streams_mat_[i]));
  // }
  // for (int ei = 0; ei < rois.size(); ++ei) {
  //     cudaStreamSynchronize(cuda_streams_[ei]);
  // }
  // TIMEREND(ROI_PREPROCESS_TIME);
  // DURATION_ms(ROI_PREPROCESS_TIME);

  // TIMERSTART(ENQUENE_TIME);
  context_->enqueueV2(&buffs_[0], stream_, nullptr);
  CUDACHECK(cudaMemcpyAsync(num_dets_, buffs_[1], out_size_[0] * sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_boxes_, buffs_[2], out_size_[1] * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_scores_, buffs_[3],
                            out_size_[2] * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaMemcpyAsync(det_classes_, buffs_[4], out_size_[3] * sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);
  // TIMEREND(EffInfer);
  // DURATION_ms(EffInfer);

  return true;
}

bool YOLOv8Infer::InitIO() {
  int numBinds = engine_->getNbBindings();
  for (int i = 0; i < numBinds; ++i) {
    auto t_dim = engine_->getBindingDimensions(i);
    for (int j = 0; j < t_dim.nbDims; ++j) {
      if (i == 0) {
        in_size_ *= t_dim.d[j];
      } else {
        out_size_[i - 1] *= t_dim.d[j];
      }
    }
  }
  // CUDACHECK(cudaMallocHost(&in_data_, in_size_ * sizeof(float)));
  // num_dets = realloc(num_dets, this->out_size[0] * sizeof(int));
  // det_boxes = realloc(det_boxes, this->out_size[1] * sizeof(float));
  // det_scores = realloc(num_dets, this->out_size[2] * sizeof(float));
  // det_classes = realloc(num_dets, this->out_size[3] * sizeof(int));
  // in_data_ = new float[in_size_];
  num_dets_ = new int[out_size_[0]];
  det_boxes_ = new float[out_size_[1]];
  det_scores_ = new float[out_size_[2]];
  det_classes_ = new int[out_size_[3]];
  CUDACHECK(cudaMalloc(&buffs_[0], in_size_ * sizeof(float)));
  CUDACHECK(cudaMalloc(&buffs_[1], out_size_[0] * sizeof(int)));
  CUDACHECK(cudaMalloc(&buffs_[2], out_size_[1] * sizeof(float)));
  CUDACHECK(cudaMalloc(&buffs_[3], out_size_[2] * sizeof(float)));
  CUDACHECK(cudaMalloc(&buffs_[4], out_size_[3] * sizeof(int)));
  return true;
}

bool YOLOv8Infer::NMSPlugin() {
  auto previous_output = network_->getOutput(0);
  network_->unmarkOutput(*previous_output);

  nvinfer1::Dims3 *strides = new nvinfer1::Dims3(1, 1, 1);
  nvinfer1::Dims3 *starts = new nvinfer1::Dims3(0, 0, 0);
  auto add_shuffle = network_->addShuffle(*previous_output);
  add_shuffle->setSecondTranspose({0, 2, 1});

  auto bs = add_shuffle->getOutput(0)->getDimensions().d[0];
  auto numBoxes = add_shuffle->getOutput(0)->getDimensions().d[1];
  auto temp = add_shuffle->getOutput(0)->getDimensions().d[2];
  std::cout << "bs :" << bs << ","
            << "numBoxes: " << numBoxes << ","
            << "temp:" << temp << std::endl;
  nvinfer1::Dims3 *shapes = new nvinfer1::Dims3(bs, numBoxes, 4);
  auto boxes = network_->addSlice(*add_shuffle->getOutput(0), *starts, *shapes,
                                  *strides);
  numBoxes = temp - 4;
  starts->d[2] = 4;
  shapes->d[2] = numBoxes;
  auto scores = network_->addSlice(*add_shuffle->getOutput(0), *starts, *shapes,
                                   *strides);

  /*
          "plugin_version": "1",
          "background_class" : -1, # no background class
          "max_output_boxes" : detections_per_img,
          "score_threshold" : score_thresh,
          "iou_threshold" : nms_thresh,
          "score_activation" : False,
          "box_coding" : 1,
  */
  /*
        new parameter:
        "plugin_version": "1",
        "background_class": -1,  # no background class
        "max_output_boxes": detections_per_img,
        "score_threshold": score_thresh,
        "iou_threshold": nms_thresh,
        "score_activation": False,
        "box_coding": 1,
  */
  auto registry =
      nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kDEFAULT);
  auto creator = registry->getPluginCreator("EfficientNMS_TRT", "1");
  std::vector<nvinfer1::PluginField> fc;
  int pdata1[] = {-1};
  int pdata2[] = {100};
  float pdata3[] = {0.5};
  float pdata4[] = {0.5};
  int pdata5[] = {1};
  int pdata6[] = {0};
  fc.push_back(nvinfer1::PluginField("background_class", pdata1,
                                     nvinfer1::PluginFieldType::kINT32));
  fc.push_back(nvinfer1::PluginField("max_output_boxes", pdata2,
                                     nvinfer1::PluginFieldType::kINT32));
  fc.push_back(nvinfer1::PluginField("score_threshold", pdata3,
                                     nvinfer1::PluginFieldType::kFLOAT32));
  fc.push_back(nvinfer1::PluginField("iou_threshold", pdata4,
                                     nvinfer1::PluginFieldType::kFLOAT32));
  fc.push_back(nvinfer1::PluginField("box_coding", pdata5,
                                     nvinfer1::PluginFieldType::kINT32));
  fc.push_back(nvinfer1::PluginField("score_activation", pdata6,
                                     nvinfer1::PluginFieldType::kINT32));
  auto fc2 = new nvinfer1::PluginFieldCollection();
  fc2->fields = fc.data();
  fc2->nbFields = fc.size();
  auto nmsLayer = creator->createPlugin("nms_layer", fc2);
  std::vector<nvinfer1::ITensor *> layerInputs = {boxes->getOutput(0),
                                                  scores->getOutput(0)};
  auto layer = network_->addPluginV2(layerInputs.data(), 2, *nmsLayer);
  layer->getOutput(0)->setName("num");
  layer->getOutput(1)->setName("boxes");
  layer->getOutput(2)->setName("scores");
  layer->getOutput(3)->setName("classes");
  for (int i = 0; i < 4; ++i) {
    network_->markOutput(*layer->getOutput(i));
  }
  nmsLayer->destroy();
  return true;
}

float YOLOv8Infer::D_letterbox(cv::cuda::GpuMat image,
                               cv::cuda::GpuMat &out_image,
                               cv::cuda::Stream stream,
                               const cv::Size &new_shape, int stride,
                               const cv::Scalar &color, bool fixed_shape,
                               bool scale_up) {
  cv::Size shape = image.size();
  float r = std::min((float)new_shape.height / (float)shape.height,
                     (float)new_shape.width / (float)shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }
  int newUnpad[2]{(int)std::round((float)shape.width * r),
                  (int)std::round((float)shape.height * r)};

  cv::cuda::resize(image, image, cv::Size(newUnpad[0], newUnpad[1]), 0.0, 0.0,
                   1, stream);

  float dw = new_shape.width - newUnpad[0];
  float dh = new_shape.height - newUnpad[1];

  if (!fixed_shape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));

  cv::cuda::copyMakeBorder(image, out_image, top, bottom, left, right,
                           cv::BORDER_CONSTANT, color, stream);

  return 1.0f / r;
}

void YOLOv8Infer::D_Preprocess(cv::cuda::GpuMat &image,
                               cv::cuda::Stream stream) {
  scale_ = D_letterbox(image, image, stream, {det_width_, det_height_}, 32,
                       {114, 114, 114}, true);
  cv::cuda::cvtColor(image, image, cv::COLOR_BGR2RGB, 0, stream);
}

bool YOLOv8Infer::BlobFromImg(const cv::Mat &img, float *blob) {
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] =
            (((float)img.at<cv::Vec3b>(h, w)[c] / 255.0) - means_[c]) /
            stds_[c];
      }
    }
  }
  return true;
}
