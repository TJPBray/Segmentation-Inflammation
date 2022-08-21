%% inflammationDemo_v1

%Run script following importInflammationData



% 5. Augment the data

%5.1 Create data augmenter object (specifying specific augmentations)
% imageAugmenter = imageDataAugmenter( ...  % Dummy augmenter to ensure
% that new datastore functions as expected
%     'RandRotation',[0,0])

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30])


% % %5.2 Perform data augmentation for combined data store
% auTrainingDS = transform(trainingDS,@classificationAugmentationPipeline, ...
%     IncludeInfo=true);
% 
% %Also apply to imds in isolation to allow preview
% auTrainingDsPreview = transform(imds,@classificationAugmentationPipeline, ...
%     IncludeInfo=true);

auTrainingDS = transform(trainingDS, @(data)augment(imageAugmenter,data));

auTrainingDS2 = transform(trainingDS,@segmentationAugmentationPipeline);

%5.3 Preview augmentation
figure
subplot(1,3,1)
prev=preview(trainingDS);
montage(prev{1},'DisplayRange',[])

subplot(1,3,2)
prev2=preview(auTrainingDS);
montage(prev2{1},'DisplayRange',[])

subplot(1,3,3)
prev3=preview(auTrainingDS2);
montage(prev3{1},'DisplayRange',[0 0.5])


%% 6. Create the U-Net

%6.1 Create
imageSize = size(trainingImages,1:2);
numClasses = 2;
encoderDepth=4; %Can have a deeper network than with MNIST to reflect larger images
lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth)
% lgraph = unetLayersWithDiceLoss(imageSize, numClasses,'EncoderDepth',encoderDepth)

%6.2 Display the network
plot(lgraph)

%% 7. Train the U-net
%7.1 Set training options (descent method, learning rate, epochs etc)
options = trainingOptions('sgdm', ...
    'InitialLearnRate',e-2, ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 20, ...
    'Plots','training-progress',...
    'VerboseFrequency',10);

% Note - look into how to specify loss (need Dice loss)
% https://github.com/hongweilibran/wmh_ibbmTum - look for 

%7.2 Train the network on the  dataset
net = trainNetwork(auTrainingDS,lgraph,options)


%% 7. Assess performance on training data

% prediction = net.predict(testDS);
prediction = net.predict(imds);

i=40;

img = readimage(imds,i);
labelimage = uint16(readimage(labelds,i));

figure
title('Training dataset performance')
subplot(2,2,1)
imshow(img,[])
title('Image')

subplot(2,2,2)
imshow(labelimage,[])
title('Label')

subplot(2,2,3)
imshow(prediction(:,:,1,i),[])
title('Prediction')



%% 8. Predict

%8.1 Visualise data, labels and prediction for chosen i
% prediction = net.predict(testDS);
prediction = net.predict(testImds);

i=28;

img = readimage(testImds,i);
labelimage = uint16(readimage(testLabelds,i));

figure
subplot(2,2,1)
imshow(img,[])
title('Image')

subplot(2,2,2)
imshow(labelimage,[])
title('Label')

subplot(2,2,3)
imshow(prediction(:,:,1,i),[])
title('Prediction')


%8.3 Evaluate loss on test dataset matlab (calculate Dice)
