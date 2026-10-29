
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat medianFilter5x5CUDA(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
    {
        std::cout << "CUDA cihazi bulunamadi." << std::endl;
        return cv::Mat();
    }

    try
    {
        cv::cuda::setDevice(0);

        cv::cuda::GpuMat gpuInput;
        cv::cuda::GpuMat gpuOutput;

        gpuInput.upload(input);

        // 5x5 CUDA medyan filtresini olustur
        cv::Ptr<cv::cuda::Filter> filter =
            cv::cuda::createMedianFilter(
                CV_8UC1,
                5
            );

        filter->apply(gpuInput, gpuOutput);

        cv::Mat result;
        gpuOutput.download(result);

        std::cout << "5x5 medyan filtresi CUDA ile uygulandi."
            << std::endl;

        return result;
    }
    catch (const cv::Exception& e)
    {
        std::cout << "CUDA medyan filtresi hatasi: "
            << e.what() << std::endl;
        return cv::Mat();
    }
}
