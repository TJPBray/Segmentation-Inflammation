
%Written by Tim Bray July 2022
%t.bray@ucl.ac.uk

clear all 

%% 1. Specify folders 

%
maskfolderR1 = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/TJPB Cleaned Inflammation';
maskfolderR2 = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/MHC Cleaned Segmentation';
imagefolder = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/STIR images';

%Get folder info
maskfolderR1info=dir(maskfolderR1);
maskfolderR2info=dir(maskfolderR2);
imagefolderinfo=dir(imagefolder);

%Number of dead files at start of folder
d=3;

%Specify number of real files
numFiles = (numel(maskfolderR1info)-d);

%Image size for dataset (note that a minority of datasets may need to be
%adjusted to fit this)
imageSize = 320;

%Specify max number of slices
maxSlices = 25;

%Load dataset info (which are training, test etc)
load datasetInfo.mat

%% 2. Get masks

%Prefill arrays
% maskStacks = zeros(imageSize,imageSize,maxSlices,numFiles);
% imageStacks = zeros(imageSize,imageSize,maxSlices,numFiles);

%2.1 Loop through subjects
for k=1: numFiles

%2.2 Get filename from parotid folder
filename1=maskfolderR1info(k+d).name;
filename2=maskfolderR2info(k+d).name;

%2.3 Create additional namelist for easy viewing
data.maskname1{k}=filename1;
data.maskname2{k}=filename2;

%2.4 Get cleaned segmentation for each subject, then combine into single
%mask (majority vote)
mask1 = niftiread(fullfile(maskfolderR1,filename1));
mask2 = niftiread(fullfile(maskfolderR2,filename2));

mask = mask1.*mask2;

%2.5 Permute
mask=permute(mask,[2 1 3]);
mask=flip(mask,1);

%2.6 Convert mask to uint16
mask=uint16(mask);

%2.7 Convert all nonzero values in mask to 1 (i.e. use both high and low
%thresholds)
mask(mask>0)=1;

%2.8 Determine number of slices for current stack
slices = size(mask,3);

%2.9 Resize image to target if needed
if size(mask,3)==imageSize
    ;
else 
    mask = imresize(mask, [imageSize imageSize]);
    disp('Resizing mask')
end

%2.9 Determine if image belongs to training or test dataset and export to
%trainingLabels or testLabels accordingly

if datasetInfo.testData(k)==1 

    %Option a: Export to test labels

    %Get current size
    if exist('testLabels') == 1
    currentSize = size(testLabels,3);
    else
    currentSize = 0;
    end

    %Add to stack
    testLabels(:,:,(currentSize+1):(currentSize+slices)) = mask;

    %Update id labels
    idLabels.test((currentSize+1):(currentSize+slices),1)= datasetInfo.id(k);
    treatmentLabels.test((currentSize+1):(currentSize+slices),1)= datasetInfo.treatment(k);
else

    %Option b: Export to training labels

    %Get current size
    if exist('trainingLabels') == 1
    currentSize = size(trainingLabels,3);
    else
    currentSize = 0;
    end

    %Add to stack
    trainingLabels(:,:,(currentSize+1):(currentSize+slices)) = mask;
    
    %Update id labels accordingly
    idLabels.training((currentSize+1):(currentSize+slices),1)= datasetInfo.id(k);
    treatmentLabels.training((currentSize+1):(currentSize+slices),1)= datasetInfo.treatment(k);

end

%% 3. Get images

% Get filename from parotid folder
imagefilename=imagefolderinfo(k+d).name;
    
%Add name to structure
data.imagename{k}=imagefilename;

%Get image for each subject 
image=niftiread(fullfile(imagefolder,imagefilename));

%Permute
image=permute(image,[2 1 3]);
image=flip(image,1);

%Convert image to uint16
image=uint16(image);



%2.9 Check image is correct size, else resize
if size(image,3)==imageSize
    ;
else 
    image = imresize(image, [imageSize imageSize]);
    disp('Resizing image')
end

%2.10 Determine if image belongs to training or test dataset and export to
%trainingImages or testImages accordingly

if datasetInfo.testData(k)==1 

    %Option a: Export to test images

    %Get current size
    if exist('testImages') == 1
    currentSize = size(testImages,3);
    else
    currentSize = 0;
    end

    %Add to stack
    testImages(:,:,(currentSize+1):(currentSize+slices)) = image;

else

    %Option b: Export to training images

    %Get current size
    if exist('trainingImages') == 1
    currentSize = size(trainingImages,3);
    else
    currentSize = 0;
    end

    %Add to stack
    trainingImages(:,:,(currentSize+1):(currentSize+slices)) = image;

end


%% Create overlay image

overlayimage=image+(mask*1000);

%% Display

%Set display option
display=0;

%Display if disp==1
if display==1
newanal2(overlayimage)
else ;
end

end




% %% 4. Export to hdf5 for use in Keras
% 
% %4.1 Select folder for export and set as current folder
% cd '/Users/TJB57/Dropbox/MATLAB/Segmentation Inflammation/inflammationData/hdf5/';
% 
% %4.2 Create empty hdf dataset for images and labels
% h5create('trainingImages.h5','/trainingImageDataSet',size(trainingImages))
% h5create('testImages.h5','/testImageDataSet',size(testImages))
% h5create('trainingLabels.h5','/trainingLabelsDataSet',size(trainingLabels))
% h5create('testLabels.h5','/testLabelsDataSet',size(testLabels))
% 
% %4.3 Add image data to hdf dataset
% h5write('trainingImages.h5','/trainingImageDataSet',trainingImages)
% h5write('testImages.h5','/testImageDataSet',testImages)
% h5write('trainingLabels.h5','/trainingLabelsDataSet',trainingLabels)
% h5write('testLabels.h5','/testLabelsDataSet',testLabels)
% 
% % display contents of example
% h5disp('trainingImages.h5')
% 
% %4.4 Read data from hdf5
% trainingImageCheck = h5read('trainingImages.h5','/trainingImageDataSet');
% testImageCheck = h5read('testImages.h5','/testImageDataSet');
% trainingLabelCheck = h5read('trainingLabels.h5','/trainingLabelsDataSet');
% testLabelCheck = h5read('testLabels.h5','/testLabelsDataSet');
% 
% 
% %Choose slice in combined stack to view training data with labels
% newanal2(trainingImageCheck)
% sl = 578;
% 
% figure
% subplot(1,2,1)
% imshow(trainingImageCheck(:,:,sl),[])
% subplot(1,2,2)
% imshow(trainingLabelCheck(:,:,sl),[])
% 
% %Choose slice in combined stack to view test data with labels
% newanal2(testImageCheck)
% sl = 61;
% 
% figure
% subplot(1,2,1)
% imshow(testImageCheck(:,:,sl),[])
% subplot(1,2,2)
% imshow(testLabelCheck(:,:,sl),[])
% 
% %4.5 Create id and treatment labels
% %id
% h5create('idLabelsTraining.h5','/idLabels',size(idLabels.training))
% h5create('idLabelsTest.h5','/idLabels',size(idLabels.test))
% 
% h5write('idLabelsTraining.h5','/idLabels',idLabels.training)
% h5write('idLabelsTest.h5','/idLabels',idLabels.test)
% 
% %treatment
% h5create('treatmentLabelsTraining.h5','/treatmentLabels',size(treatmentLabels.training))
% h5create('treatmentLabelsTest.h5','/treatmentLabels',size(treatmentLabels.test))
% 
% h5write('treatmentLabelsTraining.h5','/treatmentLabels',treatmentLabels.training)
% h5write('treatmentLabelsTest.h5','/treatmentLabels',treatmentLabels.test)


%% 5. Create label datastores for MATLAB use

%5.1 Select folder for splitting of training data images
imageDir = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/Datastores/trainingDataImages'
cd(imageDir)

for k=1:size(trainingImages,3)
    image=trainingImages(:,:,k);
    imwrite(image,strcat('trainingDataImage',num2str(k),'.png'));
    [1,k]
end

% Create an imageDatastore object to store the training images.
imds = imageDatastore(imageDir);

%5.2 Select folder for splitting of training data images
labelDir = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/Datastores/trainingDataLabels'
cd(labelDir)

for k=1:size(trainingLabels,3)
    image=uint8(trainingLabels(:,:,k));
    imwrite(image,strcat('trainingDataLabels',num2str(k),'.png'));
    [2,k]
end

% Define the class names and their associated label IDs.
classNames = ["inflammation", "background"];
labelIDs   = [255 0]; %Use 255 as 1s correspond to 255 in jpg files

% Create a labelDatastore object to store the training images.
labelds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%Combine labels and images into a single data store
trainingDS = combine(imds,labelds);

%5.3 Select folder for splitting of test data images
imageDir = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/Datastores/testDataImages'
cd(imageDir)

for k=1:size(testImages,3)
    image=testImages(:,:,k);
    imwrite(image,strcat('testDataImage',num2str(k),'.png'));
    [3,k]
end

% Create an imageDatastore object to store the training images.
testImds = imageDatastore(imageDir);

%5.4 Select folder for splitting of test data labels
labelDir = '/scratch0/NOT_BACKED_UP/timbray/inflammationData/Datastores/testDataLabels'
cd(labelDir)

for k=1:size(testLabels,3)
    image=uint8(testLabels(:,:,k));
    imwrite(image,strcat('testDataLabels',num2str(k),'.png'));
    [4,k]
end

% Define the class names and their associated label IDs.
classNames = ["inflammation", "background"];
labelIDs   = [255 0]; %Use 255 as 1s correspond to 255 in jpg files

% Create a labelDatastore object to store the training images.
testLabelds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%Combine labels and images into a single data store
testDS = combine(testImds,testLabelds);



