
function dataOut = imageRegressionAugmentationPipeline(dataIn)
%Based heavily on
%https://uk.mathworks.com/help/deeplearning/ug/image-augmentation-using-image-processing-toolbox.html#AugmentImagesForDeepLearningWorkflowsExample-20

dataOut = cell([size(dataIn,1),2]);
for idx = 1:size(dataIn,1)
    
%Get data
inputImage=dataIn{idx,1};
targetImage=dataIn{idx,2};

%     % Resize images to 32-by-32 pixels and convert to data type single
%     inputImage = im2single(imresize(dataIn{idx,1},[32 32]));
%     targetImage = im2single(uint16(imresize(dataIn{idx,2},[32 32])));
    
%     % Add salt and pepper noise
%     inputImage = imnoise(inputImage,"salt & pepper");
    
    % Normalise
    inputImage = double(inputImage) / double(max(inputImage,[],'all'));

    % Imadjust
    inputImage=imadjust(inputImage,[0 0.5])

    % Add randomized rotation and scale
    tform = randomAffine2d(Scale=[0.9,1.1],Rotation=[-30 30]);
    outputView = affineOutputView(size(inputImage),tform);
    
    % Use imwarp with the same tform and outputView to augment both images
    % the same way
    inputImage = imwarp(inputImage,tform,OutputView=outputView);
    targetImage = imwarp(targetImage,tform,OutputView=outputView);
    
    dataOut(idx,:) = {inputImage,targetImage};
end

end