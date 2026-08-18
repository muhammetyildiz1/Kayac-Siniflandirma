# Kayaç Sınıflandırma - Derin Öğrenme ile

Bu proje, kayaç görüntülerini derin öğrenme yöntemleri kullanarak sınıflandırmayı amaçlamaktadır. Çalışmada hem literatürde yaygın kullanılan klasik CNN mimarileri hem de bu proje kapsamında geliştirilen özgün bir model (**MYNet**) karşılaştırmalı olarak değerlendirilmiştir.

## 📌 Proje Özeti

- **Veri Seti:** 35 farklı kayaç sınıfını içeren geniş bir görüntü veri seti
- **Kullanılan Modeller:**
  - MYNet (bu çalışmada geliştirilen özgün model)
  - AlexNet
  - VGG16
  - VGG19
  - SqueezeNet
  - ResNet50
  - GoogleNet
  - DarkNet19
- **Ortam:** MATLAB (Deep Learning Toolbox)
- **Değerlendirme Metrikleri:** Precision, Recall, F1-Score (her sınıf için ayrı ayrı ve genel olarak)

## 📁 Klasör Yapısı

```
kayac-siniflandirma/
├── mynet/            # Özgün model kodları
├── alexnet/
├── vgg16/
├── vgg19/
├── squeezenet/
├── resnet50/
├── googlenet/
├── darknet19/
├── veriseti/         # Veri seti (varsa, boyut büyükse .gitignore'a ekleyin)
├── bildiri/          # Çalışmaya ait bildiri/rapor dosyaları
└── README.md
```

## 🚀 Nasıl Çalıştırılır

1. MATLAB R2023b (veya üzeri) ve **Deep Learning Toolbox** kurulu olmalıdır.
2. Kullanmak istediğiniz modelin klasörüne girin (örn. `mynet/`).
3. İlgili `.m` dosyasını MATLAB'de açıp çalıştırın.
4. Eğitim tamamlandığında Command Window'da her sınıf için **Precision / Recall / F1-Score** metrikleri otomatik olarak raporlanır.

## 📊 Sonuçlar

Modellerin performans karşılaştırması ve detaylı metrikler `bildiri/` klasöründeki rapor dosyasında yer almaktadır.

## 👤 Geliştirici

Muhammet
