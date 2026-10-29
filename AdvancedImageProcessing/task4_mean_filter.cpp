
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

// 3x3 ortalama filtresini kendimiz uygulariz.
// Piksel satirlari birden fazla thread ile islenebilir.
cv::Mat meanFilterManual(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    cv::Mat result(input.size(), CV_8UC1);

    const int kernelSize = 3;
    const int radius = kernelSize / 2;

    unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    threadCount = std::min(
        threadCount,
        static_cast<unsigned int>(input.rows));

    auto processRows = [&](int startRow, int endRow)
        {
            for (int y = startRow; y < endRow; y++)
            {
                for (int x = 0; x < input.cols; x++)
                {
                    int sum = 0;
                    int count = 0;

                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ny = y + ky;
                            int nx = x + kx;

                            // Goruntu sinirlarini kontrol et
                            if (ny >= 0 && ny < input.rows &&
                                nx >= 0 && nx < input.cols)
                            {
                                sum += input.at<uchar>(ny, nx);
                                count++;
                            }
                        }
                    }

                    result.at<uchar>(y, x) =
                        static_cast<uchar>(
                            (sum + count / 2) / count);
                }
            }
        };

    std::vector<std::thread> threads;
    int rowsPerThread =
        (input.rows + static_cast<int>(threadCount) - 1) /
        static_cast<int>(threadCount);

    for (unsigned int i = 0; i < threadCount; i++)
    {
        int startRow = static_cast<int>(i) * rowsPerThread;
        int endRow = std::min(
            startRow + rowsPerThread, input.rows);

        if (startRow < input.rows)
        {
            threads.emplace_back(
                processRows, startRow, endRow);
        }
    }

    for (auto& thread : threads)
        thread.join();

    return result;
}
