%% Data downloaded from https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/
% (Preformatted for MATLAB)

%% 1. Import data 

load("mnist.mat")

%To check if the dataset is properly loaded, you can display a digit with the following command:
exampleIm=(training.images(:,:,18)*255);

% image(exampleIm);

% Since the image is stored with values between 0 and 1, you have to scale the value between 0 and 255. 
% The Matlab function rescale is dedicated to this purpose.
imshow(training.images(:,:,18),[]);

% Since the label are stored in the dataset, it becomes easy to get the label (or digit) associated to an image:
training.labels(18)

%% 2. Pad arrays to make compatible with U-net

%2.1 For training images
training.images=padarray(training.images,[2 2],0,'both');

%2.2 For test images
test.images=padarray(test.images,[2 2],0,'both');

%% 3. Create a modified set of labels for whole image by thresholding

%3.1 For training dataset
training.modLabels = training.images>0.5;
imshow(training.modLabels(:,:,18),[]);

%3.2 For test dataset
test.modLabels = test.images>0.5;
figure, imshow(test.modLabels(:,:,29),[])

%% 4. Save files and generate datastore format
%Pick number of datasets to use 
n=200;

%4.1 Load training data images
imageDir = '/Users/tjb57/Dropbox/MATLAB/Segmentation/MNISTMATLAB/trainingData/'
cd(imageDir)

for k=1:n
    image=training.images(:,:,k);
    imwrite(image,strcat('trainingDataImage',num2str(k),'.jpg'));
end

% Create an imageDatastore object to store the training images.
imds = imageDatastore(imageDir);

%4.2 Load training data labels
labelDir = '/Users/tjb57/Dropbox/MATLAB/Segmentation/MNISTMATLAB/testData/';
cd(labelDir)

for k=1:n
    labelImage = training.modLabels(:,:,k);
    imwrite(labelImage,strcat('trainingDataLabel',num2str(k),'.jpg'));
end

% Define the class names and their associated label IDs.
classNames = ["digit", "background"];
labelIDs   = [255 0]; %Use 255 as 1s correspond to 255 in jpg files

% Create a labelDatastore object to store the training images.
labelds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%Combine labels and images into a single data store
ds = combine(imds,labelds);


%4.3 Load test data images
imageDir = '/Users/tjb57/Dropbox/MATLAB/Segmentation/MNISTMATLAB/testData/'
cd(imageDir)

for k=1:n
    image=test.images(:,:,k);
    imwrite(image,strcat('testDataImage',num2str(k),'.jpg'));
end

% Create an imageDatastore object to store the training images.
testImds = imageDatastore(imageDir);

%4.4 Load test data labels
labelDir = '/Users/tjb57/Dropbox/MATLAB/Segmentation/MNISTMATLAB/testData/';
cd(labelDir)

for k=1:n
    labelImage = test.modLabels(:,:,k);
    imwrite(labelImage,strcat('testDataLabel',num2str(k),'.jpg'));
end

% Define the class names and their associated label IDs.
classNames = ["digit", "background"];
labelIDs   = [255 0]; %Use 255 as 1s correspond to 255 in jpg files

% Create a labelDatastore object to store the training images.
testLabelds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%Combine labels and images into a single data store
testDs = combine(testImds,testLabelds);


%% 5. Create the U-Net

%5.1 Create
imageSize = [32 32];
numClasses = 2;
lgraph = unetLayersWithDiceLoss(imageSize, numClasses)

%5.2 Display the network
plot(lgraph)

%% 6. Train the U-net
%6.1 Set training options (descent method, learning rate, epochs etc)
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 10, ...
    'Plots','training-progress',...
    'VerboseFrequency',10);

% Note - look into how to specify loss (need Dice loss)
% https://github.com/hongweilibran/wmh_ibbmTum - look for 

%6.2 Train the network
net = trainNetwork(ds,lgraph,options)

%% 7. Predict

%4.1 Get prediction
prediction = net.predict(testImds);

%4.2 Visualise data, labels and prediction for chosen i
i=15;
img = readimage(testImds,i);
labelimage = readimage(testLabelds,i);

figure
subplot(2,2,1)
imshow(img,[])
title('Image')

subplot(2,2,2)
imshow(double(labelimage),[])
title('Label')

subplot(2,2,3)
imshow(prediction(:,:,1,i),[])
title('Prediction')
