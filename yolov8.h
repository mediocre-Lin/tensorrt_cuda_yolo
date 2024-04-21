#ifndef TRT_YOLOV8_H
#define TRT_YOLOV8_H
#include "trt_infer.h"
class YOLOv8Infer : public BaseInfer {
 public:
  bool SetUp(const std::string& model_path, const int work_space,
             const bool fp16 , const int batch_num,const int det_width, const int det_height,const std::vector<float> means, const std::vector<float> stds);
  bool Infer(const std::vector<cv::Mat>& mat);
  bool Infer(const cv::cuda::GpuMat& d_mat, const std::vector<cv::Rect>& rois);
  bool Infer(const std::vector<cv::Rect>& rois);
  void LoadData(const cv::Mat& data);
  void LoadData(const cv::cuda::HostMem& data);

  bool GetRes(std::vector<DetPredictorOutput>& res);
  cudaStream_t GetCudaStream();
  ~YOLOv8Infer();
  cv::cuda::GpuMat big_d_mat;
  cudaStream_t stream_ = nullptr;

 private:
  bool NavieInfer(const std::vector<cv::Mat>& mat);
  bool EffInfer(const std::vector<cv::Mat>& mat);
  bool EffInfer(const std::vector<cv::Rect>& rois);
  bool EffInfer(const cv::cuda::GpuMat& d_mat, const std::vector<cv::Rect>& rois);

  bool InitIO();
  bool NMSPlugin();
  bool BatchPreprocess(const std::vector<cv::Mat>& mat);
  float D_letterbox(cv::cuda::GpuMat image, cv::cuda::GpuMat& out_image,cv::cuda::Stream stream,
                    const cv::Size& new_shape = cv::Size(640, 640),
                    int stride = 32,
                    const cv::Scalar& color = cv::Scalar(114, 114, 114),
                    bool fixed_shape = false, bool scale_up = true);
  void D_Preprocess(cv::cuda::GpuMat& image,cv::cuda::Stream stream);
  bool BlobFromImg(const cv::Mat& img, float* blob);
  nvinfer1::IExecutionContext* context_ = nullptr;
  cv::cuda::Stream stream_mat_;
  std::vector<cudaStream_t> cuda_streams_;

  std::vector<cv::cuda::GpuMat> d_mats_;
  int batch_num_= 1;
  int det_width_;
  int det_height_;
  std::vector<float> means_;
  std::vector<float> stds_;
  int ori_width_;
  int ori_height_;
  float scale_ = 1.0;
  int in_size_ = 1;
  size_t pitch_;
  uchar3 *d_mat_data_ = nullptr;
  std::vector<int> out_size_{1, 1, 1, 1};
  int *h_x = nullptr;
  int *h_y = nullptr;
  int *h_w = nullptr;
  int *h_h = nullptr;

  int *d_x = nullptr;
  int *d_y = nullptr;
  int *d_w = nullptr;
  int *d_h = nullptr;
	float *d_means = nullptr;
	float *d_stds = nullptr;
  int* num_dets_ = nullptr;
  float* det_boxes_ = nullptr;
  float* det_scores_ = nullptr;
  int* det_classes_ = nullptr;
  float* in_data_ = nullptr;
  void* buffs_[5];
};
#endif
