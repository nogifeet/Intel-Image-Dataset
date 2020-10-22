train_dir='D:\Repositories\Projects\Image Classification\Intel Dataset\Data\seg_train\seg_train';
val_dir='D:\Repositories\Projects\Image Classification\Intel Dataset\Data\seg_test\seg_test';

train_imds=imageDatastore(train_dir,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
val_imds=imageDatastore(val_dir,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

%check train and validation labels
train_lbl=train_imds.Labels;
val_lbl=val_imds.Labels;

%imshow(preview(train_imds));
%imshow(preview(val_imds));

imageSize=[224 224 3];
train_augmenter = imageDataAugmenter('RandRotation',[-20,20],...
    'RandXTranslation',[-3,3],...
    'RandYTranslation',[-3,3]);


val_augmenter=imageDataAugmenter();

train = augmentedImageDatastore(imageSize,train_imds,'DataAugmentation',train_augmenter);
validation = augmentedImageDatastore(imageSize,val_imds,'DataAugmentation',val_augmenter);


%minibatch = preview(train);
%imshow(imtile(minibatch.input));

%net=vgg19;
model=layers_1;

opts = trainingOptions('sgdm', ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.1,...
    'ValidationData',validation,...
    'MiniBatchSize',32,...
    'ExecutionEnvironment','gpu');

%net = trainNetwork(train,model,opts);

%make prediction on test data
cd 'D:\Repositories\Projects\Image Classification\Intel Dataset\Data\seg_pred\seg_pred'
img = imread("30.jpg");
img_1 = imresize(img,[224,224]);
testpreds = classify(net,img_1);
testpreds

intel_net = net;
save('intel_vgg_19','intel_net');



