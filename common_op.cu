#include"opencv2/core/cuda.hpp"
#include"opencv2/cudaarithm.hpp"
#include"opencv2/cudawarping.hpp"
#include"opencv2/cudaimgproc.hpp"
#include"opencv2/core/cuda_stream_accessor.hpp"
#include"opencv2/opencv.hpp"
#include <cuda.h>

#include"cuda_runtime_api.h"
#include"cuda_device_runtime_api.h"
#include"device_launch_parameters.h"
__global__ void Batch_ROI_BGR2RGB_NORM_Kernel_2(const uchar3* in,float* out,int batch,int* roi_x, int* roi_y, int* roi_width, int* roi_height, float* means, float* stds )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x ;
    const int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if((x >=  roi_width[0]) || (y >=  roi_height[0]))
        return;
    for(int i = 0; i < batch; ++i){
        uchar3 v = in[y+ roi_y[i], x+ roi_x[i]];
        int step = roi_width[0] * roi_height[0];
        int idx = i* step  + y * roi_width[0] + x;
        out[idx] = ((v.z / 255.f) - means[0]) / stds[0];  
        out[idx + step] =((v.y / 255.f) - means[1]) / stds[1];
        out[idx + 2 * step] =   ((v.x / 255.f) - means[2]) / stds[2];
    }


}

__global__ void Batch_ROI_BGR2RGB_NORM_Kernel(cv::cuda::PtrStepSz<uchar3> in,float* out,int batch,int* roi_x, int* roi_y, int* roi_width, int* roi_height, float* means, float* stds )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x ;
    const int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if((x >=  roi_width[0]) || (y >=  roi_height[0]))
        return;
    for(int i = 0; i < batch; ++i){
        uchar3 v = in(y+ roi_y[i], x+ roi_x[i]);
        int step = roi_width[0] * roi_height[0];
        int idx = i* step  + y * roi_width[0] + x;
        out[idx] = ((v.z / 255.f) - means[0]) / stds[0];  
        out[idx + step] =((v.y / 255.f) - means[1]) / stds[1];
        out[idx + 2 * step] =   ((v.x / 255.f) - means[2]) / stds[2];
    }


}

__global__ void ROI_BGR2RGB_NORM_Kernel(cv::cuda::PtrStepSz<uchar3> in,float* out,int roi_x, int roi_y, int roi_width, int roi_height, float* means, float* stds )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x ;
    const int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if((x >=  roi_width) || (y >=  roi_height))
        return;
    uchar3 v = in(y+ roi_y, x+ roi_x);
    int step = roi_width * roi_height;
    int idx = y * roi_width + x;
    out[idx] = ((v.z / 255.f) - means[0]) / stds[0];  
    out[idx + step] =((v.y / 255.f) - means[1]) / stds[1];
    out[idx + 2 * step] =   ((v.x / 255.f) - means[2]) / stds[2];

}

__global__ void toNCHW_Norm_Kernel(cv::cuda::PtrStepSz<uchar3> in, float* out,float* means, float* stds )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= in.cols) || (y >= in.rows))
        return;
    uchar3 v = in(y, x);
    int step = in.cols * in.rows;
    int idx = y * in.cols + x;
    out[idx] = ((v.x / 255.f) - means[0]) / stds[0];
    out[idx + step] =((v.y / 255.f) - means[1]) / stds[1];
    out[idx + 2 * step] = ((v.z / 255.f) - means[2]) / stds[2];

}

extern "C" void toNCHW_Norm(const cv::cuda::GpuMat& input, float* output,float* means, float* stds,
            cudaStream_t stream)
{
    const dim3 threads(32, 8);
	const dim3 grid((input.cols +  threads.x - 1) / threads.x, (input.rows + threads.y - 1) / threads.y);
	float *d_means = nullptr;
	float *d_stds = nullptr;
	cudaMalloc(&d_means, 3 * sizeof(float));
	cudaMalloc(&d_stds, 3 * sizeof(float));
	cudaMemcpyAsync(d_means, means, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_stds, stds, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
    toNCHW_Norm_Kernel<<<grid,threads,0,stream>>>(input, output,d_means,d_stds);
	cudaFree(d_means);
	cudaFree(d_stds);

}

extern "C" void ROI_BGR2RGB_NORM(const cv::cuda::GpuMat& input, float* output, const float* means, const float* stds,int roi_x, int roi_y, int roi_width, int roi_height,
            cudaStream_t stream)
{
    const dim3 threads(32, 8);
	const dim3 grid((roi_width +  threads.x - 1) / threads.x, (roi_height + threads.y - 1) / threads.y);
	float *d_means = nullptr;
	float *d_stds = nullptr;
	cudaMalloc(&d_means, 3 * sizeof(float));
	cudaMalloc(&d_stds, 3 * sizeof(float));
	cudaMemcpyAsync(d_means, means, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_stds, stds, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
    ROI_BGR2RGB_NORM_Kernel<<<grid,threads,0,stream>>>(input, output,roi_x, roi_y, roi_width, roi_height,d_means,d_stds);
	cudaFree(d_means);
	cudaFree(d_stds);
}
extern "C" void Batch_ROI_BGR2RGB_NORM(const cv::cuda::GpuMat& input, float* output, int batch,const float* means, const float* stds,const std::vector<cv::Rect> &rois,
            const cudaStream_t stream)
{
    const dim3 threads(32, 8);
	const dim3 grid((rois[0].width +  threads.x - 1) / threads.x, (rois[0].height + threads.y - 1) / threads.y);
    int *h_x = new int[rois.size()];
    int *h_y = new int[rois.size()];
    int *h_w = new int[rois.size()];
    int *h_h = new int[rois.size()];
    for(int i =0; i <rois.size();++i){
        h_y[i] = rois[i].y;
        h_x[i] = rois[i].x;
        h_w[i] = rois[i].width;
        h_h[i] = rois[i].height;
    }
    int *d_x = nullptr;
    int *d_y = nullptr;
    int *d_w = nullptr;
    int *d_h = nullptr;
	float *d_means = nullptr;
	float *d_stds = nullptr;
	cudaMalloc(&d_means, 3 * sizeof(float));
	cudaMalloc(&d_stds, 3 * sizeof(float));
	cudaMalloc(&d_x, rois.size() * sizeof(int));
	cudaMalloc(&d_y, rois.size() * sizeof(int));
	cudaMalloc(&d_w, rois.size() * sizeof(int));
	cudaMalloc(&d_h, rois.size() * sizeof(int));
	cudaMemcpyAsync(d_x, h_x, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_y, h_y, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_w, h_w, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_h, h_h, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_means, means, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_stds, stds, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
    Batch_ROI_BGR2RGB_NORM_Kernel<<<grid,threads,0,stream>>>(input, output,batch,d_x, d_y, d_w, d_h,d_means,d_stds);
	cudaFree(d_stds);
	cudaFree(d_means);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_w);
	cudaFree(d_h);
    delete h_x;
    delete h_y;
    delete h_w;
    delete h_h;
}

extern "C" void Batch_ROI_BGR2RGB_NORM_2(const uchar3* input, float* output, int batch,const float* means, const float* stds,const std::vector<cv::Rect> &rois,
            cudaStream_t stream)
{
    const dim3 threads(32, 8);
	const dim3 grid((rois[0].width +  threads.x - 1) / threads.x, (rois[0].height + threads.y - 1) / threads.y);
    int *h_x = new int[rois.size()];
    int *h_y = new int[rois.size()];
    int *h_w = new int[rois.size()];
    int *h_h = new int[rois.size()];
    for(int i =0; i <rois.size();++i){
        h_y[i] = rois[i].y;
        h_x[i] = rois[i].x;
        h_w[i] = rois[i].width;
        h_h[i] = rois[i].height;
    }
    int *d_x = nullptr;
    int *d_y = nullptr;
    int *d_w = nullptr;
    int *d_h = nullptr;
	float *d_means = nullptr;
	float *d_stds = nullptr;
	cudaMalloc(&d_means, 3 * sizeof(float));
	cudaMalloc(&d_stds, 3 * sizeof(float));
	cudaMalloc(&d_x, rois.size() * sizeof(int));
	cudaMalloc(&d_y, rois.size() * sizeof(int));
	cudaMalloc(&d_w, rois.size() * sizeof(int));
	cudaMalloc(&d_h, rois.size() * sizeof(int));
	cudaMemcpyAsync(d_x, h_x, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_y, h_y, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_w, h_w, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_h, h_h, rois.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_means, means, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_stds, stds, 3 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
    Batch_ROI_BGR2RGB_NORM_Kernel_2<<<grid,threads,0,stream>>>(input, output,batch,d_x, d_y, d_w, d_h,d_means,d_stds);
	cudaFree(d_stds);
	cudaFree(d_means);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_w);
	cudaFree(d_h);
    delete h_x;
    delete h_y;
    delete h_w;
    delete h_h;


}