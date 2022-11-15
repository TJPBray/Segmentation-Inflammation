%U-Net Demo
%Demonstration of uNet implementation using Matlab toolboxes
%Requires imaging processing, deep learning and computer vision toolboxes
%Written by T Bray April 2022
%t.bray@ucl.ac.uk
%Based on documentation at 
% https://uk.mathworks.com/help/vision/ref/unetlayers.html

% v2 runs a basic 2D unet with the addition of data augmentation 

%% 1. Load the data and create the U-net
% 1.1 Load training images and pixel labels into the workspace.
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

% 1.2 Create an imageDatastore object to store the training images.

% (i) As imdatastore structure
imds = imageDatastore(imageDir);

% (ii) As volume
imdsCell = readall(imds);

for k=1:size(imdsCell,1)
imdsVol(:,:,k) = imdsCell{k,1};
end
    
%1.3 Define the class names and their associated label IDs.
classNames = ["triangle","background"];
labelIDs   = [255 0];

%1.4 Create a pixelLabelDatastore object to store the ground truth pixel labels for the training images.
% (i) As imdatastore structure
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

% (ii) As volume
pxdsCell = readall(pxds);

for k=1:size(pxdsCell,1)
pxdsVol(:,:,k) = pxdsCell{k,1};
end
    

%1.5 Combine labels and images into a single data store including both
%images and labels
ds = combine(imds,pxds);

%% 2. Augment the data

%2.1 Create data augmenter object (specifying specific augmentations)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3])

%2.2 Perform data augmentation for combined data store
% 
ds = pixelLabelImageDatastore(imds,pxds);

auds = transform(ds, @(data)augment(imageAugmenter,data));

imageSize = [32 32];

%2.3 Preview augmentation


%% 2. Create the U-Net
%2.1 Create
imageSize = [32 32];
numClasses = 2;
lgraph = unetLayers(imageSize, numClasses)

%2.2 Display the network
plot(lgraph)

%% 3. Train the U-net
%3.1 Set training options (descent method, learning rate, epochs etc)
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 20, ...
    'Plots','training-progress',...
    'VerboseFrequency',10);

%3.2 Train the network
net = trainNetwork(ds,lgraph,options)

%% 4. Use the trained network to predict outputs for imds (note not independent test set for this toy example) 

%4.1 Get prediction
prediction = net.predict(ds);

%4.2 Visualise data, labels and prediction for chosen i
i=5;
img = readimage(imds,i);
labelImg = readimage(pxds,i);

i2=20;
img2 = readimage(imds,i2);
labelImg2 = readimage(pxds,i2);

figure
subplot(2,3,1)
imshow(img,[])
title('Image 1')

subplot(2,3,2)
imshow(double(labelImg),[])
title('Label 1')

subplot(2,3,3)
imshow(prediction(:,:,1,i),[])
title('Prediction 1')

subplot(2,3,4)
imshow(img2,[])
title('Image 2')

subplot(2,3,5)
imshow(double(labelImg2),[])
title('Label 2')

subplot(2,3,6)
imshow(prediction(:,:,1,i2),[])
title('Prediction 2')
