
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>

// 1. Tuz-biber gurultusu olustur
cv::Mat addSaltPepperNoise(const cv::Mat& input)
{
    cv::Mat noisy = input.clone();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> probability(0.0, 1.0);

    // Piksellerin %5'ine gurultu ekle
    for (int y = 0; y < noisy.rows; y++)
    {
        for (int x = 0; x < noisy.cols; x++)
        {
            double randomValue = probability(rng);

            if (randomValue < 0.025)
                noisy.at<uchar>(y, x) = 0;       // Biber
            else if (randomValue > 0.975)
                noisy.at<uchar>(y, x) = 255;     // Tuz
        }
    }

    return noisy;
}

// 2. Kendi yazdigimiz 5x5 medyan filtresi
cv::Mat medianFilter5x5Manual(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    cv::Mat result(input.size(), CV_8UC1);

    const int radius = 2;

    unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    threadCount = std::min(
        threadCount,
        static_cast<unsigned int>(input.rows));

    auto processRows = [&](int startRow, int endRow)
        {
            std::vector<uchar> neighbors;
            neighbors.reserve(25);

            for (int y = startRow; y < endRow; y++)
            {
                for (int x = 0; x < input.cols; x++)
                {
                    neighbors.clear();

                    // 5x5 komsuluktaki pikselleri topla
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ny = y + ky;
                            int nx = x + kx;

                            if (ny >= 0 && ny < input.rows &&
                                nx >= 0 && nx < input.cols)
                            {
                                neighbors.push_back(
                                    input.at<uchar>(ny, nx));
                            }
                        }
                    }

                    // Ortadaki degeri bulmak icin sirala
                    std::sort(neighbors.begin(), neighbors.end());

                    result.at<uchar>(y, x) =
                        neighbors[neighbors.size() / 2];
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

// 3. OpenCV'nin hazir 5x5 medyan filtresi
cv::Mat medianFilter5x5OpenCV(const cv::Mat& input)
{
    cv::Mat result;
    cv::medianBlur(input, result, 5);
    return result;
}
