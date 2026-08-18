clear all;

egitim_yolu = 'C:\Users\Muhammet\Desktop\kayac_siniflandirma\veriseti\train'; 
test_yolu = 'C:\Users\Muhammet\Desktop\kayac_siniflandirma\veriseti\test';

egitim_veriseti = imageDatastore(egitim_yolu, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

test_veriseti = imageDatastore(test_yolu, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

dogru_siralama = string(1:35); 
egitim_veriseti.Labels = reordercats(egitim_veriseti.Labels, dogru_siralama);
test_veriseti.Labels = reordercats(test_veriseti.Labels, dogru_siralama);

etiketler = egitim_veriseti.Labels;
kategori_sayisi = numel(categories(etiketler));

net = alexnet;
lgraph = layerGraph(net);
inputSize = [227 227 3];

egitim_veriseti_aug = augmentedImageDatastore(inputSize, egitim_veriseti);
test_veriseti_aug = augmentedImageDatastore(inputSize, test_veriseti);

new_fc = fullyConnectedLayer(kategori_sayisi, ...
    'Name', 'new_fc8', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, 'fc8', new_fc);

new_softmax = softmaxLayer('Name', 'new_softmax');
lgraph = replaceLayer(lgraph, 'prob', new_softmax);

new_output = classificationLayer('Name', 'new_output');
lgraph = replaceLayer(lgraph, 'output', new_output);

options = trainingOptions("sgdm", ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[kayac_siniflandirma_alexnet_model, info] = trainNetwork(egitim_veriseti_aug, lgraph, options);

tahmin = classify(kayac_siniflandirma_alexnet_model, test_veriseti_aug);
dogruluk_orani = sum(tahmin == test_veriseti.Labels) / numel(test_veriseti.Labels) * 100;
disp(['Accuracy (Doğruluk Orani): %', num2str(dogruluk_orani)]);

figure;
confusionchart(test_veriseti.Labels, tahmin);
title('Confusion Matrix - AlexNet');

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

save('C:\Users\Muhammet\Desktop\kayac_siniflandirma\alexnet\kayac_siniflandirma_alexnet_egitim.mat', ...
    'kayac_siniflandirma_alexnet_model');