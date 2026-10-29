
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

// 1. OpenCV'nin hazir CLAHE yontemi
cv::Mat claheOpenCV(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    if (input.empty())
        return cv::Mat();

    cv::Mat result;

    cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(2.0, cv::Size(8, 8));

    clahe->apply(input, result);

    return result;
}

// 2. Manuel ve cok is parcacikli CLAHE
cv::Mat claheManual(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);

    if (input.empty())
        return cv::Mat();

    const int tileRows = 8;
    const int tileCols = 8;
    const int bins = 256;
    const double clipLimit = 2.0;
    const int tileCount = tileRows * tileCols;

    cv::Mat result(input.size(), CV_8UC1);

    // Bolge sinirlari esit dagitilir; son bolgeler
    // kalan pikselleri de kapsar.
    std::vector<int> xStart(tileCols), xEnd(tileCols);
    std::vector<int> yStart(tileRows), yEnd(tileRows);

    for (int tx = 0; tx < tileCols; tx++)
    {
        xStart[tx] = tx * input.cols / tileCols;
        xEnd[tx] = (tx + 1) * input.cols / tileCols;
    }

    for (int ty = 0; ty < tileRows; ty++)
    {
        yStart[ty] = ty * input.rows / tileRows;
        yEnd[ty] = (ty + 1) * input.rows / tileRows;
    }

    std::vector<std::vector<uchar>> lookup(
        tileCount, std::vector<uchar>(bins, 0));

    unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    threadCount = std::min(
        threadCount, static_cast<unsigned int>(tileCount));

    // Histogrami kirp, tasan degerleri dagit ve LUT olustur.
    auto processTile = [&](int tileIndex)
        {
            int ty = tileIndex / tileCols;
            int tx = tileIndex % tileCols;

            int x0 = xStart[tx];
            int x1 = xEnd[tx];
            int y0 = yStart[ty];
            int y1 = yEnd[ty];

            int width = x1 - x0;
            int height = y1 - y0;
            int pixelCount = width * height;

            if (pixelCount <= 0)
                return;

            std::vector<int> histogram(bins, 0);

            for (int y = y0; y < y1; y++)
            {
                const uchar* row = input.ptr<uchar>(y);

                for (int x = x0; x < x1; x++)
                    histogram[row[x]]++;
            }

            // CLAHE clip limit
            int limit = std::max(
                1,
                static_cast<int>(
                    clipLimit * pixelCount / bins));

            int excess = 0;

            for (int i = 0; i < bins; i++)
            {
                if (histogram[i] > limit)
                {
                    excess += histogram[i] - limit;
                    histogram[i] = limit;
                }
            }

            // Tasani histogram kutularina esit dagit.
            // Her dagitimdan sonra siniri asan kutulari
            // yeniden kirparak tasani bir sonraki tura aktar.
            while (excess > 0)
            {
                int eligibleBins = 0;

                for (int i = 0; i < bins; i++)
                {
                    if (histogram[i] < limit)
                        eligibleBins++;
                }

                if (eligibleBins == 0)
                {
                    // Tum kutular limitteyse kalan miktari
                    // kontrollu sekilde dagit.
                    for (int i = 0; i < bins && excess > 0; i++)
                    {
                        histogram[i]++;
                        excess--;
                    }
                    continue;
                }

                int share = std::max(1, excess / eligibleBins);
                int distributed = 0;

                for (int i = 0; i < bins && excess > 0; i++)
                {
                    int capacity = limit - histogram[i];

                    if (capacity <= 0)
                        continue;

                    int add = std::min(
                        capacity, std::min(share, excess));

                    histogram[i] += add;
                    excess -= add;
                    distributed += add;
                }

                if (distributed == 0)
                    break;
            }

            // CDF ve LUT
            int cumulative = 0;

            for (int i = 0; i < bins; i++)
            {
                cumulative += histogram[i];

                int value = static_cast<int>(
                    255.0 * cumulative / pixelCount + 0.5);

                lookup[tileIndex][i] =
                    static_cast<uchar>(
                        std::clamp(value, 0, 255));
            }
        };

    // Tile hesaplamalarini sinirli sayida thread ile yap.
    std::vector<std::thread> threads;

    for (int i = 0; i < tileCount; i++)
    {
        threads.emplace_back(processTile, i);

        if (threads.size() >= threadCount)
        {
            for (auto& thread : threads)
                thread.join();

            threads.clear();
        }
    }

    for (auto& thread : threads)
        thread.join();

    // Pikselleri bolgeler arasinda bilineer enterpolasyonla isle.
    auto processRows = [&](int startRow, int endRow)
        {
            for (int y = startRow; y < endRow; y++)
            {
                double gy =
                    (static_cast<double>(y) + 0.5) *
                    tileRows / input.rows - 0.5;

                int tyBase = static_cast<int>(std::floor(gy));
                double fy = gy - tyBase;

                int ty0 = std::clamp(tyBase, 0, tileRows - 1);
                int ty1 = std::clamp(tyBase + 1, 0, tileRows - 1);
                fy = std::clamp(fy, 0.0, 1.0);

                const uchar* source = input.ptr<uchar>(y);
                uchar* destination = result.ptr<uchar>(y);

                for (int x = 0; x < input.cols; x++)
                {
                    double gx =
                        (static_cast<double>(x) + 0.5) *
                        tileCols / input.cols - 0.5;

                    int txBase = static_cast<int>(std::floor(gx));
                    double fx = gx - txBase;

                    int tx0 = std::clamp(txBase, 0, tileCols - 1);
                    int tx1 = std::clamp(txBase + 1, 0, tileCols - 1);
                    fx = std::clamp(fx, 0.0, 1.0);

                    int pixel = source[x];

                    double top =
                        lookup[ty0 * tileCols + tx0][pixel] * (1.0 - fx) +
                        lookup[ty0 * tileCols + tx1][pixel] * fx;

                    double bottom =
                        lookup[ty1 * tileCols + tx0][pixel] * (1.0 - fx) +
                        lookup[ty1 * tileCols + tx1][pixel] * fx;

                    double value =
                        top * (1.0 - fy) + bottom * fy;

                    destination[x] = static_cast<uchar>(
                        std::clamp(
                            static_cast<int>(value + 0.5),
                            0, 255));
                }
            }
        };

    threads.clear();

    int rowsPerThread =
        (input.rows + static_cast<int>(threadCount) - 1) /
        static_cast<int>(threadCount);

    for (unsigned int i = 0; i < threadCount; i++)
    {
        int startRow = static_cast<int>(i) * rowsPerThread;
        int endRow = std::min(
            startRow + rowsPerThread, input.rows);

        if (startRow < input.rows)
            threads.emplace_back(processRows, startRow, endRow);
    }

    for (auto& thread : threads)
        thread.join();

    std::cout << "Manuel CLAHE cok is parcacikli calisti."
        << std::endl;

    return result;
}
