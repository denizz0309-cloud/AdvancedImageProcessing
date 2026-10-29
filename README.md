# Gelişmiş Görüntü İşleme

Bu proje, OpenCV ve C++ kullanılarak temel görüntü işleme tekniklerinin manuel, çok iş parçacıklı ve CUDA destekli uygulamalarını içerir.

## Proje İçeriği

### Görev 1: Manuel Kontrast İyileştirme
- Gri tonlamalı görüntüde minimum ve maksimum piksel değerlerinin bulunması.
- Piksel değerlerinin 0–255 aralığına doğrusal olarak ölçeklenmesi.

### Görev 2: Manuel Histogram Eşitleme
- 256 seviyeli histogramın hesaplanması.
- Kümülatif dağılım fonksiyonunun (CDF) oluşturulması.
- Histogram eşitlemenin manuel uygulanması.

### Görev 3: CLAHE
- OpenCV ile CLAHE uygulanması.
- CLAHE algoritmasının manuel uygulanması ve sonuçların karşılaştırılması.

### Görev 4: 3×3 Ortalama Filtresi
- Ortalama filtresinin manuel uygulanması.
- Çok iş parçacıklı ve CUDA destekli sürümünün kullanılması.

### Görev 5: 5×5 Medyan Filtresi
- Görüntüye tuz-biber gürültüsü eklenmesi.
- OpenCV ve manuel medyan filtrelerinin uygulanması.
- CUDA destekli medyan filtre sonucunun elde edilmesi.

## Kullanılan Teknolojiler
- C++
- OpenCV
- CUDA
- Çok iş parçacıklı programlama (Multithreading)
- Visual Studio 2022

## Proje Dosyaları
- `main.cpp`: Ana program.
- `task1_contrast.cpp`: Manuel kontrast iyileştirme.
- `task2_histogram.cpp`: Manuel histogram eşitleme.
- `task3_clahe.cpp`: CLAHE uygulamaları.
- `task4_mean_filter.cpp`: Manuel ortalama filtresi.
- `task4_mean_filter_cuda.cpp`: CUDA ortalama filtresi.
- `task5_median_filter.cpp`: Tuz-biber gürültüsü ve medyan filtreleri.
- `task5_median_filter_cuda.cpp`: CUDA medyan filtresi.
- `input.jpg`: Giriş görüntüsü.
- `Teslim_Sonuclari/`: İşlem sonuçlarını içeren görseller.

## Çalıştırma
1. `AdvancedImageProcessing.sln` dosyasını Visual Studio 2022 ile açın.
2. OpenCV kurulum yollarını ve gerekli kütüphaneleri kendi sisteminize göre yapılandırın.
3. CUDA destekli işlemler için uyumlu CUDA destekli OpenCV kurulumu ve NVIDIA GPU gerekir.
4. Projeyi derleyip çalıştırın.

**Not:** CUDA işlemlerinin kullanılabilirliği, OpenCV derlemesine ve donanım yapılandırmasına bağlıdır.
