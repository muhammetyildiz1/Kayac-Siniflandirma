clear all;

egitim_yolu = 'C:\Users\Muhammet\Desktop\kayac_siniflandirma\veriseti\train'; 
test_yolu   = 'C:\Users\Muhammet\Desktop\kayac_siniflandirma\veriseti\test';

egitim_veriseti = imageDatastore(egitim_yolu, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
test_veriseti   = imageDatastore(test_yolu, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

etiketler = egitim_veriseti.Labels;
kategori_sayisi = numel(categories(etiketler)); 

inputSize = [224 224 3];

layers = [
    imageInputLayer(inputSize, 'Name', 'giris')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relufc')
    dropoutLayer(0.5, 'Name', 'd1')
    
    fullyConnectedLayer(kategori_sayisi, 'Name', 'cikis')
    softmaxLayer('Name', 'soft')
    classificationLayer('Name', 'sinif')
];

egitim_veriseti_aug = augmentedImageDatastore(inputSize, egitim_veriseti);
test_veriseti_aug   = augmentedImageDatastore(inputSize, test_veriseti);

options = trainingOptions('adam', ...
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.1, ...        
    'InitialLearnRate', 0.0001, ...         
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'ValidationData', test_veriseti_aug, ... 
    'ValidationFrequency', 400, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[kayac_siniflandirma_mynet_model, info] = trainNetwork(egitim_veriseti_aug, layers, options);

tahmin = classify(kayac_siniflandirma_mynet_model, test_veriseti_aug);
dogruluk_orani = sum(tahmin == test_veriseti.Labels) / numel(test_veriseti.Labels) * 100;
disp(['Accuracy (Doğruluk Orani): %', num2str(dogruluk_orani)]);

figure;
confusionchart(test_veriseti.Labels, tahmin);
title('Confusion Matrix - MYNet');

cm = confusionmat(test_veriseti.Labels, tahmin);

numClasses = kategori_sayisi;
precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1score   = zeros(numClasses,1);

for i = 1:numClasses
    TP = cm(i,i);
    FP = sum(cm(:,i)) - TP;
    FN = sum(cm(i,:)) - TP;
    precision(i) = TP / (TP + FP + eps);
    recall(i)    = TP / (TP + FN + eps);
    f1score(i)   = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i)+eps);
end

macro_precision = mean(precision);
macro_recall    = mean(recall);
macro_f1        = mean(f1score);

disp('Değerlendirme Metrikleri');
disp(table(categories(etiketler), precision, recall, f1score, ...
    'VariableNames', {'Sınıf','Precision','Recall','F1Score'}));

fprintf('Doğruluk Oranı  : %.2f%%\n', dogruluk_orani);
fprintf('Macro Precision : %.2f\n', macro_precision*100);
fprintf('Macro Recall    : %.2f\n', macro_recall*100);
fprintf('Macro F1-score  : %.2f\n', macro_f1*100);

save('C:\Users\Muhammet\Desktop\kayac_siniflandirma\mynet\kayac_siniflandirma_mynet_egitim.mat', ...
    'kayac_siniflandirma_mynet_model');