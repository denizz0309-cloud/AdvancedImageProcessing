
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <limits>

cv::Mat linearStretchManual(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    if (input.empty())
        return cv::Mat();

    unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    threadCount = std::min(
        threadCount,
        static_cast<unsigned int>(input.rows));

    // 1. Her is parcacigi icin ayri minimum ve maksimum
    std::vector<int> localMin(threadCount, 255);
    std::vector<int> localMax(threadCount, 0);
    std::vector<std::thread> threads;

    auto findMinMax = [&](int startRow, int endRow, unsigned int id)
        {
            int minValue = 255;
            int maxValue = 0;

            for (int y = startRow; y < endRow; y++)
            {
                const uchar* row = input.ptr<uchar>(y);

                for (int x = 0; x < input.cols; x++)
                {
                    int pixel = row[x];

                    if (pixel < minValue)
                        minValue = pixel;

                    if (pixel > maxValue)
                        maxValue = pixel;
                }
            }

            localMin[id] = minValue;
            localMax[id] = maxValue;
        };

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
                findMinMax, startRow, endRow, i);
        }
    }

    for (auto& thread : threads)
        thread.join();

    int minValue = 255;
    int maxValue = 0;

    for (unsigned int i = 0; i < threadCount; i++)
    {
        minValue = std::min(minValue, localMin[i]);
        maxValue = std::max(maxValue, localMax[i]);
    }

    std::cout << "Minimum piksel degeri: "
        << minValue << std::endl;
    std::cout << "Maksimum piksel degeri: "
        << maxValue << std::endl;
    std::cout << "Minimum/maksimum hesabi cok is parcacikli yapildi."
        << std::endl;

    cv::Mat result(input.size(), CV_8UC1);

    if (maxValue == minValue)
    {
        result.setTo(0);
        return result;
    }

    // 2. Kontrasti paralel olarak 0-255 araligina genislet
    threads.clear();

    auto stretchRows = [&](int startRow, int endRow)
        {
            for (int y = startRow; y < endRow; y++)
            {
                const uchar* source = input.ptr<uchar>(y);
                uchar* destination = result.ptr<uchar>(y);

                for (int x = 0; x < input.cols; x++)
                {
                    double scaled =
                        (source[x] - minValue) * 255.0 /
                        (maxValue - minValue);

                    int value = static_cast<int>(scaled + 0.5);
                    value = std::clamp(value, 0, 255);

                    destination[x] = static_cast<uchar>(value);
                }
            }
        };

    for (unsigned int i = 0; i < threadCount; i++)
    {
        int startRow = static_cast<int>(i) * rowsPerThread;
        int endRow = std::min(
            startRow + rowsPerThread, input.rows);

        if (startRow < input.rows)
        {
            threads.emplace_back(
                stretchRows, startRow, endRow);
        }
    }

    for (auto& thread : threads)
        thread.join();

    std::cout << "Kontrast iyilestirme cok is parcacikli yapildi."
        << std::endl;

    return result;
}
