
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat meanFilterCUDA(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    // CUDA cihazi yoksa hata vermeden bildir
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
    {
        std::cout << "CUDA cihazi bulunamadi." << std::endl;
        return cv::Mat();
    }

    try
    {
        cv::cuda::setDevice(0);

        // Goruntuyu GPU belleðine aktar
        cv::cuda::GpuMat gpuInput;
        cv::cuda::GpuMat gpuOutput;
        gpuInput.upload(input);

        // 3x3 ortalama filtresi olustur
        cv::Ptr<cv::cuda::Filter> filter =
            cv::cuda::createBoxFilter(
                CV_8UC1,
                CV_8UC1,
                cv::Size(3, 3)
            );

        // Filtreyi GPU uzerinde uygula
        filter->apply(gpuInput, gpuOutput);

        // Sonucu CPU belleðine geri al
        cv::Mat result;
        gpuOutput.download(result);

        std::cout << "3x3 ortalama filtresi CUDA ile uygulandi."
            << std::endl;

        return result;
    }
    catch (const cv::Exception& e)
    {
        std::cout << "CUDA filtre hatasi: "
            << e.what() << std::endl;
        return cv::Mat();
    }
}
