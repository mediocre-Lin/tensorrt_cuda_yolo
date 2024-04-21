#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/opencv.hpp"
#include "thread_pool.h"
#include "yolov7.h"
#include "yolov8.h"
// 相机取图
void captureImage(int id, std::vector<int> &recorder) {
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(850));
    recorder[id] += 1;
    std::cout << "Camera " << id << " captured image."
              << "image length : " << recorder[id] << std::endl;
  }
}

// 检测工位
void detectImage(int id, int batch_num, YOLOv8Infer &det,
                 const std::vector<cv::cuda::HostMem> &mem,
                 std::vector<int> &recorder, bool warm_up) {
  if (warm_up) {
    for (int i = 0; i < 5; ++i) {
      TIMERSTART(WARMUP_TIME);
      det.LoadData(mem[id]);

      cv::Size blockSize(1024, 800);

      std::vector<std::vector<cv::Rect>> roi_set;
      std::vector<std::vector<DetPredictorOutput>> res;
      int roi_set_num =
          8000 / blockSize.height * 51200 / blockSize.width / batch_num;
      roi_set.resize(roi_set_num);
      res.resize(roi_set_num);
      for (int i = 0; i < roi_set_num; ++i) {
        roi_set[i].resize(batch_num);
      }

      int count = 0;
      for (int y = 0; y < 8000; y += blockSize.height) {
        for (int x = 0; x < 51200; x += blockSize.width) {
          cv::Rect roi(x, y, blockSize.width, blockSize.height);
          roi.width = std::min(roi.width, 8000 - x);
          roi.height = std::min(roi.height, 51200 - y);
          roi_set[count / batch_num][count % batch_num] = roi;
          count++;
        }
      }
      // #pragma omp parallel for
      // det.big_d_mat.upload(mem[id],cv::cuda::StreamAccessor::wrapStream(det.stream_));

      for (int i = 0; i < roi_set_num; ++i) {
        det.Infer(roi_set[i]);
        det.GetRes(res[i]);
      }

      TIMEREND(WARMUP_TIME);
      DURATION_ms(WARMUP_TIME);
    }

  } else {
    while (true) {
      if (recorder[id] <= 0) {
        continue;
      }
      //计时
      TIMERSTART(costTime);
      // det.LoadData(mem[id]);

      cv::Size blockSize(1024, 800);

      std::vector<std::vector<cv::Rect>> roi_set;
      std::vector<std::vector<DetPredictorOutput>> res;
      int roi_set_num =
          8000 / blockSize.height * 51200 / blockSize.width / batch_num;
      roi_set.resize(roi_set_num);
      res.resize(roi_set_num);
      for (int i = 0; i < roi_set_num; ++i) {
        roi_set[i].resize(batch_num);
      }

      int count = 0;
      for (int y = 0; y < 8000; y += blockSize.height) {
        for (int x = 0; x < 51200; x += blockSize.width) {
          cv::Rect roi(x, y, blockSize.width, blockSize.height);
          roi.width = std::min(roi.width, 51200 - x);
          roi.height = std::min(roi.height, 8000 - y);
          roi_set[count / batch_num][count % batch_num] = roi;
          count++;
        }
      }
      // #pragma omp parallel for
      // det.big_d_mat.upload(mem[id],cv::cuda::StreamAccessor::wrapStream(det.stream_));

      for (int i = 0; i < roi_set_num; ++i) {
        det.Infer(roi_set[i]);
        det.GetRes(res[i]);
      }
      //计时
      TIMEREND(costTime);
      DURATION_ms(costTime);
      recorder[id] -= 1;
      std::cout << "det " << id << ": image need to process : " << recorder[id]
                << std::endl;
    }
  }
}

void getGPUInfo() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "deviceOverlap: " << prop.deviceOverlap << std::endl;
}
int main() {
  getGPUInfo();
  cudaSetDeviceFlags(cudaDeviceMapHost);
  int batch_num = 50;
  int detector_num = 3;
  int camera_num = 3;
  std::vector<std::thread> cameraThreads;
  std::vector<std::thread> detectorThreads;
  std::vector<cv::cuda::HostMem> mems;
  std::vector<int> recorder;
  std::vector<YOLOv8Infer> model_set;
  recorder.resize(camera_num, 0);
  model_set.resize(detector_num);
  mems.resize(camera_num);
  for (int i = 0; i < detector_num; ++i) {
    model_set[i].SetUp("model.trt", 32, true, batch_num, 1024, 800, {0, 0, 0},
                       {1, 1, 1});
  }
  for (int i = 0; i < camera_num; ++i) {
    std::cout << "HostMem" << std::endl;
    mems[i] = cv::cuda::HostMem(cv::Mat::zeros(8000, 51200, CV_8UC3),
                                cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    std::cout << "HostMem End" << std::endl;
  }                                                                                                   

  // warm up
  for (int i = 0; i < detector_num; ++i) {
    detectImage(i, batch_num, model_set[i], mems,
                                 std::ref(recorder), true);
  }


  for (int i = 0; i < detector_num; ++i) {
    detectorThreads.emplace_back(detectImage, i, batch_num, model_set[i], mems,
                                  std::ref(recorder), false);
  }
  for (int i = 0; i < camera_num; ++i) {
    cameraThreads.emplace_back(captureImage, i, std::ref(recorder));
  }

  for (auto &thread : detectorThreads) {
    thread.join();
  }
  for (auto &thread : cameraThreads) {
    thread.join();
  }

  return 0;
}