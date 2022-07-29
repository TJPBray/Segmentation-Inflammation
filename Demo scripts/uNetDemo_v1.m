%U-Net Demo
%Demonstration of uNet implementation using Matlab toolboxes
%Requires imaging processing, deep learning and computer vision toolboxes
%Written by T Bray April 2022
%t.bray@ucl.ac.uk
%Based on documentation at 
% https://uk.mathworks.com/help/vision/ref/unetlayers.html

% v1 runs the basic uNet on MATLAB-provided training data
% No augmentation is used

%% 1. Load the data and create the U-net
% 1.1 Load training images and pixel labels into the workspace.
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

% 1.2 Create an imageDatastore object to store the training images.
imds = imageDatastore(imageDir);

%1.3 Define the class names and their associated label IDs.
classNames = ["triangle","background"];
labelIDs   = [255 0];

%1.4 Create a pixelLabelDatastore object to store the ground truth pixel labels for the training images.
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%1.5 Combine labels and images into a single data store
ds = combine(imds,pxds);


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
    'MiniBatchSize', 10, ...
    'Plots','training-progress',...
    'VerboseFrequency',10);

%3.2 Train the network
net = trainNetwork(ds,lgraph,options)

%% 4. Use the trained network to predict outputs for imds (note not independent test set for this toy example) 

%4.1 Get prediction
prediction = net.predict(imds);

%4.2 Visualise data, labels and prediction for chosen i
i=13;
img = readimage(imds,i);
labelImg = readimage(pxds,i)

figure
subplot(2,2,1)
imshow(img,[])
title('Image')

subplot(2,2,2)
imshow(double(labelImg),[])
title('Label')

subplot(2,2,3)
imshow(prediction(:,:,1,i),[])
title('Prediction')

