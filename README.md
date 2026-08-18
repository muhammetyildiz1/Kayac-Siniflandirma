# Kayaç Sınıflandırma

Kayaç görüntülerinin derin öğrenme ile sınıflandırılmasına yönelik bir çalışma. 35 farklı kayaç sınıfı içeren bir veri seti üzerinde, literatürde bilinen CNN mimarileri ile birlikte MYNet adlı özgün bir model eğitilip karşılaştırılmıştır.

## Kullanılan modeller

- MYNet (bu çalışma için geliştirilen model)
- AlexNet
- VGG16
- VGG19
- SqueezeNet
- ResNet50
- GoogleNet
- DarkNet19

Ortam olarak MATLAB kullanılmıştır.

## Klasör yapısı

Her model kendi klasöründe, eğitim kodu ve değerlendirme sonuçlarıyla birlikte yer alıyor:

```
mynet/
alexnet/
vgg16/
vgg19/
squeezenet/
resnet50/
googlenet/
darknet19/
```

Veri seti boyutu büyük olduğu için repoya dahil edilmemiştir: [Drill Core Image Dataset](https://huggingface.co/datasets/168sir/drill-core-image-dataset)

## Çalıştırma

1. MATLAB'de Deep Learning Toolbox kurulu olmalı.
2. İlgili model klasörüne girip `.m` dosyasını çalıştırmak yeterli.
3. Eğitim bittikten sonra Command Window'da her sınıf için precision, recall ve F1-score değerleri yazdırılıyor.

## Sonuçlar

Model bazlı değerlendirme metrikleri her klasördeki `degerlendirme_metrikleri.txt` dosyasında yer alıyor.
