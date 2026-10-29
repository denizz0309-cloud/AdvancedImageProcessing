
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

// Gorev 1: Manuel kontrast iyilestirme
cv::Mat linearStretchManual(const cv::Mat& input);

// Gorev 2: Manuel histogram esitleme
cv::Mat histogramEqualizationManual(const cv::Mat& input);

// Gorev 3: CLAHE
cv::Mat claheOpenCV(const cv::Mat& input);
cv::Mat claheManual(const cv::Mat& input);

// Gorev 4: Manuel ve CUDA destekli 3x3 ortalama filtresi
cv::Mat meanFilterManual(const cv::Mat& input);
cv::Mat meanFilterCUDA(const cv::Mat& input);

// Gorev 5: Tuz-biber gurultusu ve 5x5 medyan filtresi
cv::Mat addSaltPepperNoise(const cv::Mat& input);
cv::Mat medianFilter5x5Manual(const cv::Mat& input);
cv::Mat medianFilter5x5OpenCV(const cv::Mat& input);
cv::Mat medianFilter5x5CUDA(const cv::Mat& input);

int main()
{
    // 1. CUDA destegini kontrol et
    int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();

    std::cout << "==============================" << std::endl;
    std::cout << "CUDA destekli cihaz sayisi: "
        << cudaDevices << std::endl;

    if (cudaDevices > 0)
    {
        cv::cuda::setDevice(0);
        std::cout << "CUDA cihazi algilandi." << std::endl;
    }
    else
    {
        std::cout << "CUDA cihazi kullanilamiyor." << std::endl;
    }

    std::cout << "==============================" << std::endl;

    // 2. Goruntuyu gri tonlamali olarak oku
    cv::Mat original = cv::imread(
        "input.jpg",
        cv::IMREAD_GRAYSCALE
    );

    if (original.empty())
    {
        std::cout << "HATA: input.jpg bulunamadi!" << std::endl;
        return -1;
    }

    // 3. Dusuk kontrastli goruntu olustur
    cv::Mat lowContrast(original.size(), CV_8UC1);

    for (int y = 0; y < original.rows; y++)
    {
        for (int x = 0; x < original.cols; x++)
        {
            int pixel = original.at<uchar>(y, x);
            int newPixel = static_cast<int>(pixel * 0.4 + 70);

            if (newPixel > 255)
                newPixel = 255;

            if (newPixel < 0)
                newPixel = 0;

            lowContrast.at<uchar>(y, x) =
                static_cast<uchar>(newPixel);
        }
    }

    // 4. Manuel kontrast iyilestirme
    cv::Mat enhanced = linearStretchManual(lowContrast);

    // 5. Manuel histogram esitleme
    cv::Mat equalized =
        histogramEqualizationManual(lowContrast);

    // 6. OpenCV CLAHE
    cv::Mat claheReady = claheOpenCV(lowContrast);

    // 7. Manuel CLAHE
    cv::Mat claheCustom = claheManual(lowContrast);

    // 8. Manuel 3x3 ortalama filtresi
    cv::Mat meanFiltered = meanFilterManual(lowContrast);

    // 9. CUDA 3x3 ortalama filtresi
    cv::Mat meanFilteredCUDA;

    if (cudaDevices > 0)
    {
        meanFilteredCUDA = meanFilterCUDA(lowContrast);

        if (!meanFilteredCUDA.empty())
        {
            std::cout << "CUDA ortalama filtresi basariyla tamamlandi."
                << std::endl;
        }
    }

    // 10. Tuz-biber gurultusu olustur
    cv::Mat noisyImage = addSaltPepperNoise(lowContrast);

    // 11. OpenCV 5x5 medyan filtresi
    cv::Mat medianOpenCV =
        medianFilter5x5OpenCV(noisyImage);

    // 12. Manuel 5x5 medyan filtresi
    cv::Mat medianManual =
        medianFilter5x5Manual(noisyImage);

    // 13. CUDA 5x5 medyan filtresi
    cv::Mat medianCUDA;

    if (cudaDevices > 0)
    {
        medianCUDA = medianFilter5x5CUDA(noisyImage);

        if (!medianCUDA.empty())
        {
            std::cout << "CUDA medyan filtresi basariyla tamamlandi."
                << std::endl;
        }
        else
        {
            std::cout << "CUDA medyan filtresi sonuc uretemedi."
                << std::endl;
        }
    }

    // 14. Sonuclari dosyalara kaydet
    cv::imwrite("01_low_contrast.png", lowContrast);
    cv::imwrite("02_linear_stretch_manual.png", enhanced);
    cv::imwrite("03_histogram_equalized_manual.png", equalized);
    cv::imwrite("04_clahe_opencv.png", claheReady);
    cv::imwrite("05_clahe_manual.png", claheCustom);
    cv::imwrite("06_mean_filter_3x3_manual.png", meanFiltered);
    cv::imwrite("07_salt_pepper_noise.png", noisyImage);
    cv::imwrite("08_median_5x5_opencv.png", medianOpenCV);
    cv::imwrite("09_median_5x5_manual.png", medianManual);

    if (!meanFilteredCUDA.empty())
        cv::imwrite("10_mean_filter_3x3_cuda.png", meanFilteredCUDA);

    if (!medianCUDA.empty())
        cv::imwrite("11_median_5x5_cuda.png", medianCUDA);

    // 15. Sonuclari ekranda goster
    cv::imshow("Dusuk Kontrastli Goruntu", lowContrast);
    cv::imshow("Manuel Kontrast Iyilestirme", enhanced);
    cv::imshow("Manuel Histogram Esitleme", equalized);
    cv::imshow("OpenCV CLAHE", claheReady);
    cv::imshow("Manuel CLAHE", claheCustom);
    cv::imshow("Manuel 3x3 Ortalama Filtresi", meanFiltered);
    cv::imshow("Tuz-Biber Gurultusu", noisyImage);
    cv::imshow("OpenCV 5x5 Medyan Filtresi", medianOpenCV);
    cv::imshow("Manuel 5x5 Medyan Filtresi", medianManual);

    if (!meanFilteredCUDA.empty())
        cv::imshow("CUDA 3x3 Ortalama Filtresi", meanFilteredCUDA);

    if (!medianCUDA.empty())
        cv::imshow("CUDA 5x5 Medyan Filtresi", medianCUDA);

    std::cout << "\nTum bes gorevin islemleri tamamlandi."
        << std::endl;
    std::cout << "Sonuclar PNG dosyalari olarak kaydedildi."
        << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
