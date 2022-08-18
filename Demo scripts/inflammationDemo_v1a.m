%% inflammationDemo_v1

%Run script following importInflammationData



%% 5. Augment the data

% %5.1 Create data augmenter object (specifying specific augmentations)
% % imageAugmenter = imageDataAugmenter( ...  % Dummy augmenter to ensure
% % that new datastore functions as expected
% %     'RandRotation',[0,0])
% 
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3])
% 
% %5.2 Perform data augmentation for combined data store
% auds = transform(ds, @(data)augment(imageAugmenter,data));
% 
% %5.3 Preview augmentation


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
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 10, ...
    'Plots','training-progress',...
    'VerboseFrequency',10);

% Note - look into how to specify loss (need Dice loss)
% https://github.com/hongweilibran/wmh_ibbmTum - look for 

%7.2 Train the network on the  dataset
net = trainNetwork(trainingDS,lgraph,options)


%% 7. Assess performance on training data




%% 8. Predict

%8.1 Visualise data, labels and prediction for chosen i
i=200;
img = testImages(:,:,i);
labelImage = testLabels(:,:,i);

% prediction = net.predict(testDS);
prediction = net.predict(img);

% img = readimage(testImds,i);
% labelimage = readimage(testLabelDS,i);

figure
subplot(2,2,1)
imshow(img,[])
title('Image')

subplot(2,2,2)
imshow(labelImage,[0 1])
title('Label')

subplot(2,2,3)
imshow(prediction(:,:,1),[0 1])
title('Prediction')

%8.3 Evaluate loss on test dataset matlab (calculate Dice)
