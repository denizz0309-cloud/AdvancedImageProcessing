
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

cv::Mat histogramEqualizationManual(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    if (input.empty())
        return cv::Mat();

    unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    threadCount = std::min(
        threadCount,
        static_cast<unsigned int>(input.rows));

    // 1. Her is parcacigi icin ayri histogram
    std::vector<std::vector<int>> localHistograms(
        threadCount, std::vector<int>(256, 0));

    std::vector<std::thread> threads;

    int rowsPerThread =
        (input.rows + static_cast<int>(threadCount) - 1) /
        static_cast<int>(threadCount);

    auto calculateHistogram =
        [&](int startRow, int endRow, unsigned int id)
        {
            auto& histogram = localHistograms[id];

            for (int y = startRow; y < endRow; y++)
            {
                const uchar* row = input.ptr<uchar>(y);

                for (int x = 0; x < input.cols; x++)
                    histogram[row[x]]++;
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
                calculateHistogram, startRow, endRow, i);
        }
    }

    for (auto& thread : threads)
        thread.join();

    // 2. Yerel histogramlari birlestir
    std::vector<int> histogram(256, 0);

    for (unsigned int t = 0; t < threadCount; t++)
    {
        for (int i = 0; i < 256; i++)
            histogram[i] += localHistograms[t][i];
    }

    // 3. Kumulatif dagilim fonksiyonunu (CDF) hesapla
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];

    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + histogram[i];

    // 4. Ilk bos olmayan CDF degerini bul
    int cdfMin = 0;

    for (int i = 0; i < 256; i++)
    {
        if (cdf[i] > 0)
        {
            cdfMin = cdf[i];
            break;
        }
    }

    int totalPixels = input.rows * input.cols;
    cv::Mat result(input.size(), CV_8UC1);

    if (totalPixels == cdfMin)
    {
        input.copyTo(result);
        return result;
    }

    // 5. CDF kullanarak donusum tablosunu olustur
    std::vector<uchar> lookupTable(256, 0);

    for (int i = 0; i < 256; i++)
    {
        double mapped =
            (cdf[i] - cdfMin) * 255.0 /
            (totalPixels - cdfMin);

        int value = static_cast<int>(mapped + 0.5);
        value = std::clamp(value, 0, 255);

        lookupTable[i] = static_cast<uchar>(value);
    }

    // 6. Donusum tablosunu paralel olarak uygula
    threads.clear();

    auto applyLookup = [&](int startRow, int endRow)
        {
            for (int y = startRow; y < endRow; y++)
            {
                const uchar* source = input.ptr<uchar>(y);
                uchar* destination = result.ptr<uchar>(y);

                for (int x = 0; x < input.cols; x++)
                    destination[x] = lookupTable[source[x]];
            }
        };

    for (unsigned int i = 0; i < threadCount; i++)
    {
        int startRow = static_cast<int>(i) * rowsPerThread;
        int endRow = std::min(
            startRow + rowsPerThread, input.rows);

        if (startRow < input.rows)
            threads.emplace_back(applyLookup, startRow, endRow);
    }

    for (auto& thread : threads)
        thread.join();

    std::cout << "Histogram, CDF ve piksel donusumu "
        << "manuel ve cok is parcacikli yapildi."
        << std::endl;

    return result;
}
