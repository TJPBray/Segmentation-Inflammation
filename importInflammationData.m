
%Written by Tim Bray July 2022
%t.bray@ucl.ac.uk

%% 1. Specify folders 

maskfolderR1 = '/Users/tjb57/Dropbox/MATLAB/Segmentation/inflammationData/nifti/MasksTJPB';
maskfolderR2 = '/Users/tjb57/Dropbox/MATLAB/Segmentation/inflammationData/nifti/MasksMHC';
imagefolder = '/Users/tjb57/Dropbox/MATLAB/Segmentation/inflammationData/nifti/STIR images';

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
imageSize = 336;

%Specify max number of slices
maxSlices = 25;

%% 2. Get masks

%Prefill arrays
maskStacks = zeros(imageSize,imageSize,maxSlices,numFiles);
imageStacks = zeros(imageSize,imageSize,maxSlices,numFiles);

%2.1 Loop through subjects
for k=1: numFiles

%2.2 Get filename from parotid folder
filename=maskfolderR1info(k+d).name;

%2.3 Create additional namelist for easy viewing
data.maskname{k}=filename;

%2.4 Get initial SNAP mask for each subject (performed prior to slicer segs)
mask=niftiread(fullfile(maskfolderR1,filename));
 
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

%2.9 Add to maskStacks for export
maskStacks(:,:,1:slices,k) = mask;

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

%Add to maskStacks for export
imageStacks(:,:,1:slices,k) = image;


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


%% 4. Export to hdf5

%4.1 Select folder for export and set as current folder
cd '/Users/tjb57/Dropbox/MATLAB/Segmentation/inflammationData/hdf5/';

%4.2 Create empty hdf dataset for images and masks
h5create('imageData.h5','/imageDataSet',size(imageStacks))
h5disp('imageData.h5')

h5create('maskData.h5','/maskDataSet',size(maskStacks))

%4.3 Add image data to hdf dataset
h5write('imageData.h5','/imageDataSet',imageStacks)

h5write('maskData.h5','/maskDataSet',maskStacks)

%4.4 Read data from hdf5
imageCheck = h5read('imageData.h5','/imageDataSet');
maskCheck = h5read('maskData.h5','/maskDataSet');
