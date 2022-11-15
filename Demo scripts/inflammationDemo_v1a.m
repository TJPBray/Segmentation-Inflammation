%% inflammationDemo_v1

%Run script following importInflammationData

%% Check values in each  dataset

figure
subplot(3,2,1)
vals = (readall(dsTrain.UnderlyingDatastores{1}));
histogram(uint8([vals{:}]))
title('Training images')

subplot(3,2,2)
vals = (readall(dsTrain.UnderlyingDatastores{2}));
histogram(uint8([vals{:}]),[-0.5 0.5 1.5 2.5 3.5])
title('Training labels')
% ylim([])

subplot(3,2,3)
vals = (readall(dsValidation.UnderlyingDatastores{1}));
histogram(uint8([vals{:}]))
title('Validation images')

subplot(3,2,4)
vals = (readall(dsValidation.UnderlyingDatastores{2}));
histogram(uint8([vals{:}]),[-0.5 0.5 1.5 2.5 3.5])
title('Validation labels')
% ylim([])

subplot(3,2,5)
vals = (readall(dsTest.UnderlyingDatastores{1}));
histogram(uint8([vals{:}]))
title('Test images')

subplot(3,2,6)
vals = (readall(dsTest.UnderlyingDatastores{2}));
histogram(uint8([vals{:}]),[-0.5 0.5 1.5 2.5 3.5])
title('Test labels')
% ylim([])

%% Show examples from each dataset

% Define indices for images to display examples in each dataset
i = [211 74 10]; %For training, validation and test respectively

figure
subplot(3,2,1)
imshow(trainingImages(:,:,i(1)),[])
title('Training image')
colorbar

subplot(3,2,2)
imshow(trainingLabels(:,:,i(1)),[])
title('Training label')
colorbar

subplot(3,2,3)
imshow(validationImages(:,:,i(2)),[])
title('Validation image')
colorbar

subplot(3,2,4)
imshow(validationLabels(:,:,i(2)),[])
title('Validation label')
colorbar

subplot(3,2,5)
imshow(testImages(:,:,i(3)),[])
title('Test image')
colorbar

subplot(3,2,6)
imshow(testLabels(:,:,i(3)),[])
title('Test label')
colorbar


%% 5. Augment the data

% 5.1 Create data augmenter object (specifying specific augmentations)
% imageAugmenter = imageDataAugmenter( ...  % Dummy augmenter to ensure
% that new datastore functions as expected
%     'RandRotation',[0,0])

% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-10,10], ...
%     'RandXTranslation',[-30 30], ...
%     'RandYTranslation',[-30 30])


% %5.2 Perform data augmentation for combined data store
% auTrainingDS = transform(trainyingDS,@classificationAugmentationPipeline, ...
%     IncludeInfo=true);
% 
% %Also apply to imds in isolation to allow preview
% auTrainingDsPreview = transform(imds,@classificationAugmentationPipeline, ...
%     IncludeInfo=true);
% 
% auTrainingDS = transform(trainingDS, @(data)augment(imageAugmenter,data));

auDsTrain = transform(dsTrain,@segmentationAugmentationPipeline_WarpAndIntensityAndFlip);

% 5.3 Preview augmentation
%?Reset before preview

% prev1 = preview(trainingDS);
% prev2 = preview(auTrainingDS);
% 
% figure
% subplot(3,2,1)
% imshow(prev1{1},[0 max(prev1{1},[],'all')])
% title('Training image')
% colorbar
% 
% subplot(3,2,2)
% imshow(uint8(prev1{2}),[0 2])
% title('Training label')
% colorbar
% 
% subplot(3,2,5)
% imshow(prev2{1},[0 1])
% title('Training image')
% colorbar
% 
% subplot(3,2,6)
% imshow(uint8(prev2{2}),[0 2])
% title('Training label')
% colorbar


%
trainingDSdata = readall(dsTrain);
auDSdata = readall(auDsTrain);

figure
subplot(2,2,1)
imshow(single(trainingDSdata{i(1),1}),[0 max(single(trainingDSdata{i(1),1}),[],'all')])
title('Training image')
colorbar

subplot(2,2,2)
imshow(single(auDSdata{i(1),1}),[0 max(single(trainingDSdata{i(1),1}),[],'all')])
title('Augmented training image')
colorbar

subplot(2,2,3)
imshow(single(trainingDSdata{i(1),2}),[])
title('Training label')
colorbar

subplot(2,2,4)
imshow(single(auDSdata{i(1),2}),[])
title('Augmented training label')
colorbar

%% 6. Create the U-Net

%6.1 Specify unet architecture
imageSize = size(trainingImages,1:2);
numClasses = 2;
encoderDepth = 4; %Can have a deeper network than with MNIST to reflect larger images

%6.2 Create unet
unet = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth);

%6.3 Create an input layer without normalization (since data are preprocessed with 0 to
% 1 normalization)

inputLayer = imageInputLayer(imageSize, 'Name', 'ImageInputLayer', 'Normalization', 'None');

% replace the input layer
unet = replaceLayer(unet, 'ImageInputLayer', inputLayer);

%6.4 Create a classification output layer with class weights
outputLayer = pixelClassificationLayer('Name', 'Segmentation-Layer', 'Classes', {'background', 'foreground'}, 'ClassWeights', [0.1 0.9]);

% replace the output layer
unet = replaceLayer(unet, 'Segmentation-Layer', outputLayer);

%6.2 Display the network
% figure, plot(lgraph)

analyzeNetwork(unet)

%% 7. Train the U-net
%Principles:

%Use maximum achievable batch size depending on memory (based on nnUnet
%paper)

%Start with 'Starting Options' below and then try to iterate towards a
%better solution (results stored in Results folder)..

%'Starting Options':
% options = trainingOptions('adam', ... 
%     'InitialLearnRate',5e-4, ...
%     'L2Regularization', 1e-3, ...
%     'MaxEpochs',100, ...  
%     'MiniBatchSize',30, ...
%     'ValidationData',dsValidation,...
%     'ValidationFrequency', 25, ...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress');

% fix the random seed to ease comparison across multiple setups
rng(3);
gpurng(3);

%7.1 Set training options (descent method, learning rate, epochs etc)

options = trainingOptions('adam', ... 
    'InitialLearnRate',5e-5, ...
    'MaxEpochs',400, ...
    'MiniBatchSize',30, ...
    'ValidationData',dsValidation,...
    'ValidationFrequency', 25, ...
    'OutputNetwork', 'best-validation-loss', ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');


%Sample options 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 250, ...
%     'InitialLearnRate', 2e-5, ...
%     'L2Regularization', 1e-3, ...
%     'MiniBatchSize', 16, ...
%     'Shuffle', 'once', ...
%     'ValidationFrequency', 53, ...
%     'ValidationData',dsValidation,...
%     'Verbose', false, ...
%     'Plots', 'training-progress');

% GZ options sept 27 2022 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 250, ...
%     'InitialLearnRate', 2e-5, ...
%     'L2Regularization', 1e-3, ...
%     'MiniBatchSize', 16, ...
%     'Shuffle', 'once', ...
%     'ValidationFrequency', 53, ...
%     'ValidationData',dsValidation,...
%     'Verbose', false, ...
%     'OutputNetwork', 'best-validation-loss', ...
%     'Plots', 'training-progress');
% 
% TB old options sept 27 2022 
% options = trainingOptions('adam', ... 
%     'InitialLearnRate',5e-4, ...
%     'MaxEpochs',50, ...  
%     'MiniBatchSize',30, ...
%     'ValidationData',dsValidation,...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress');

% To halve the learning rate every 10 epochs, use 
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropPeriod',10,...
%     'LearnRateDropFactor',0.5,...
% Set the initial rate higher to deal with this

%     'ValidationPatience', 4);
%     'ValidationData',dsVal,...
%     'L2Regularization',0.005, ...
%     'VerboseFrequency',2,...
%     'Momentum',0.9, ...

%7.2 Train the network on the  dataset
[net,info] = trainNetwork(auDsTrain,unet,options)

Visualise 

results.net = net; 
results.info=info;

%% 7. Visualise performance on training and validation data

%set threshold to assign pixels as disease
inflammThresh=0.5;

%7.1 Visualise data, labels and prediction on training data
trainingPred = net.predict(dsTrain);

%Provide reference to enable identification of patient's study number
idRef = 8; 

%Select unique study id number accordingly
id = idLabels.values.training(idRef); 

%Select relevant volume for this study id (pre-treatment scans only for
%visualisation) - for both predictions and labels

imageVol = trainingImages(:,:,(idLabels.training==id & treatmentLabels.training==0));
labelVol = trainingLabels(:,:,(idLabels.training==id & treatmentLabels.training==0));
predVol = trainingPred(:,:,2,(idLabels.training==id & treatmentLabels.training==0));
predVol = squeeze(predVol);

intersection = (labelVol==1).*(predVol>inflammThresh);
union = labelVol==1 + predVol>inflammThresh;

figure
subplot(1,3,1)
montage(imageVol)
title('Images')

subplot(1,3,2)
montage(labelVol,'DisplayRange',[0 1])
title('Labels')

subplot(1,3,3)
montage(predVol, 'DisplayRange',[0 1])
title('Predictions')

%7.2 Visualise data, labels and prediction on validation data
validationPred = net.predict(dsValidation);

for k = 1:6
%Provide reference to enable identification of patient's study number
idRef = k; 

%Select unique study id number accordingly
id = idLabels.values.validation(idRef); 

%Select relevant volume for this study id (pre-treatment scans only for
%visualisation) - for both predictions and labels

imageVol = validationImages(:,:,(idLabels.validation==id & treatmentLabels.validation==0));
labelVol = validationLabels(:,:,(idLabels.validation==id & treatmentLabels.validation==0));
predVol = validationPred(:,:,2,(idLabels.validation==id & treatmentLabels.validation==0));
predVol = squeeze(predVol);

% intersection = (labelVol==1).*(predVol>inflammThresh);
% union = labelVol==1 + predVol>inflammThresh;

figure('NumberTitle','off','Name',num2str(k))

dispSlices = 5:16;

subplot(1,3,1)
montage(imageVol)

subplot(1,3,2)
labelledMontage(imageVol,labelVol,dispSlices)

subplot(1,3,3)
labelledMontage(imageVol,predVol,dispSlices)

end

%7.3 Visualise data, labels and prediction on test data
testPred = net.predict(dsTest);

for k = 1:6
%Provide reference to enable identification of patient's study number
idRef = k; 

%Select unique study id number accordingly
id = idLabels.values.test(idRef); 

%Select relevant volume for this study id (pre-treatment scans only for
%visualisation) - for both predictions and labels

imageVol = testImages(:,:,(idLabels.test==id & treatmentLabels.test==0));
labelVol = testLabels(:,:,(idLabels.test==id & treatmentLabels.test==0));
predVol = testPred(:,:,2,(idLabels.test==id & treatmentLabels.test==0));
predVol = squeeze(predVol);

% intersection = (labelVol==1).*(predVol>inflammThresh);
% union = labelVol==1 + predVol>inflammThresh;

figure('NumberTitle','off','Name',num2str(k))

dispSlices = 5:16;

subplot(1,3,1)
montage(imageVol(:,:,dispSlices))

subplot(1,3,2)
labelledMontage(imageVol,labelVol,dispSlices)

subplot(1,3,3)
labelledMontage(imageVol,predVol,dispSlices)

end



%% 8. Calculate Dice for whole datasets

%8.1 Training
% allocate the space for dice
trainingDice = zeros(numel(idLabels.values.training),2); %Two columns to allow for pre- and post-treatment scans

% allocate the space for lesion load
traininglesionLoad = zeros(numel(idLabels.values.training),2); %Two columns to allow for pre- and post-treatment scans

% loop over pre and post-treatment scans
for treatment = 1:2
    treated = treatment - 1;
    
    % for each subject, compute dice for each volume in turn
    for volumeIdx=1:numel(idLabels.values.training)
        
        sliceNumbers = (idLabels.training == idLabels.values.training(volumeIdx) & treatmentLabels.training == treated  );
        
        trainingLabelVol = trainingLabels(:,:,sliceNumbers);
        
        predictedVol = trainingPred(:,:,2,sliceNumbers);
        
        predictedVol = squeeze(predictedVol);
        
        trainingDice(volumeIdx,treatment) = dice(logical(trainingLabelVol), logical(predictedVol>inflammThresh));
%         [trainingScore(volumeIdx,treatment),trainingPrecision(volumeIdx,treatment),trainingRecall(volumeIdx,treatment)] = bfscore(logical(predictedVol>inflammThresh),logical(trainingLabelVol))
        [trainingRecall(volumeIdx,treatment),trainingPrecision(volumeIdx,treatment)] = metricsFn(logical(predictedVol>inflammThresh),logical(trainingLabelVol));

        trainingLesionLoadRef(volumeIdx, treatment) = sum(trainingLabelVol, 'all');
        trainingLesionLoad(volumeIdx, treatment) = sum(round(predictedVol), 'all'); %Rounded sum
%         trainingLesionLoad(volumeIdx, treatment) = sum(predictedVol, 'all'); %Weighted sum

    end
    
end


trainingDice

results.trainingDice = trainingDice; 
results.meanTrainingDice = mean(trainingDice,'all')

results.trainingLesionLoad = trainingLesionLoad;
results.trainingLesionLoadRef = trainingLesionLoadRef; 

results.trainingRecall = trainingRecall; 
results.meanTrainingRecall = mean(trainingRecall,'all'); 

results.trainingPrecision = trainingPrecision; 
results.meanTrainingPrecision = mean(trainingPrecision,'all'); 


figure
subplot(3,3,1)
scatter(trainingLesionLoadRef,trainingLesionLoad)
xlim([0 100000])
ylim([0 100000])
ylabel('Predictions')
xlabel('Labels')
title('Training lesion load')
legend('Pre-treatment','Post-treatment')

subplot(3,3,2)
scatter(log(trainingLesionLoadRef+1),log(trainingLesionLoad+1))
xlim([2 12])
ylim([2 12])
ylabel('log(x+1) transformed Predictions')
xlabel('log(x+1) transformed Labels')
title('Training lesion load')

subplot(3,3,3)
scatter(trainingLesionLoadRef,trainingDice)
% set(gca,'xscale','log')
xlim([0 100000])
ylim([0 1])
xlabel('Lesion load (from labels)')
ylabel('Dice')
title('Effect of lesion load on Dice - training')

hold on

%8.2 Validation
% allocate the space for dice
validationDice = zeros(numel(idLabels.values.validation),2); %Two columns to allow for pre- and post-treatment scans

% allocate the space for lesion load
validationlesionLoad = zeros(numel(idLabels.values.validation),2); %Two columns to allow for pre- and post-treatment scans

% loop over pre and post-treatment scans
for treatment = 1:2
    treated = treatment - 1;
    
    % for each subject, compute dice for each volume in turn
    for volumeIdx=1:numel(idLabels.values.validation)
        
        sliceNumbers = (idLabels.validation == idLabels.values.validation(volumeIdx) & treatmentLabels.validation == treated  );
        
        validationLabelVol = validationLabels(:,:,sliceNumbers);
        
        predictedVol = validationPred(:,:,2,sliceNumbers);
        
        predictedVol = squeeze(predictedVol);
        
        validationDice(volumeIdx,treatment) = dice(logical(validationLabelVol), logical(predictedVol>inflammThresh));
       
        [validationRecall(volumeIdx,treatment),validationPrecision(volumeIdx,treatment)] = metricsFn(logical(predictedVol>inflammThresh),logical(validationLabelVol))

        validationLesionLoadRef(volumeIdx, treatment) = sum(validationLabelVol, 'all');
        validationLesionLoad(volumeIdx, treatment) = sum(round(predictedVol), 'all');
%         validationLesionLoad(volumeIdx, treatment) = sum(predictedVol, 'all');

        

    end
end

validationDice

results.validationDice = validationDice; 
results.meanValDice = mean(validationDice,'all')

results.validationLesionLoad = validationLesionLoad;
results.validationLesionLoadRef = validationLesionLoadRef; 

results.validationRecall = validationRecall; 
results.meanValidationRecall = mean(validationRecall,'all'); 

results.validationPrecision = validationPrecision; 
results.meanValidationPrecision = mean(validationPrecision,'all'); 



subplot(3,3,4)
scatter(validationLesionLoadRef,validationLesionLoad)
ylabel('Predictions')
xlabel('Labels')
title('Validation lesion load')
xlim([0 100000])
ylim([0 100000])

subplot(3,3,5)
scatter(log(validationLesionLoadRef+1),log(validationLesionLoad+1))
xlim([2 12])
ylim([2 12])
ylabel('log(x+1) transformed Predictions')
xlabel('log(x+1) transformed Labels')
title('Validation lesion load')

subplot(3,3,6)
scatter(validationLesionLoadRef,validationDice)
xlim([0 100000])
ylim([0 1])
xlabel('Lesion load (from labels)')
ylabel('Dice')
title('Effect of lesion load on Dice - validation')


%8.3 Test
% allocate the space for dice
testDice = zeros(numel(idLabels.values.test),2); %Two columns to allow for pre- and post-treatment scans

% allocate the space for lesion load
testLesionLoad = zeros(numel(idLabels.values.test),2); %Two columns to allow for pre- and post-treatment scans

% loop over pre and post-treatment scans
for treatment = 1:2
    treated = treatment - 1;
    
    % for each subject, compute dice for each volume in turn
    for volumeIdx=1:numel(idLabels.values.test)
        
        sliceNumbers = (idLabels.test == idLabels.values.test(volumeIdx) & treatmentLabels.test == treated);
        
        testLabelVol = testLabels(:,:,sliceNumbers);
        
        predictedVol = testPred(:,:,2,sliceNumbers);
        
        predictedVol = squeeze(predictedVol);
        
        testDice(volumeIdx,treatment) = dice(logical(testLabelVol), logical(predictedVol>inflammThresh));
        
        [testRecall(volumeIdx,treatment),testPrecision(volumeIdx,treatment)] = metricsFn(logical(predictedVol>inflammThresh),logical(testLabelVol));

        testLesionLoadRef(volumeIdx, treatment) = sum(testLabelVol, 'all');
        testLesionLoad(volumeIdx, treatment) = sum(round(predictedVol), 'all');
%         validationLesionLoad(volumeIdx, treatment) = sum(predictedVol, 'all');

        

    end
end

testDice

results.testDice = testDice; 
results.meanTestDice = mean(testDice,'all');
results.medianTestDice = median(testDice,'all')

results.testLesionLoad = testLesionLoad;
results.testLesionLoadRef = testLesionLoadRef; 

results.testRecall = testRecall; 
results.meanTestRecall = mean(testRecall,'all'); 
results.medianTestRecall = median(testRecall,'all');

results.testPrecision = testPrecision; 
results.meanTestPrecision = mean(testPrecision,'all'); 
results.medianTestPrecision = median(testPrecision,'all');

subplot(3,3,7)
scatter(testLesionLoadRef,testLesionLoad)
ylabel('Predictions')
xlabel('Labels')
title('Test lesion load')
xlim([0 100000])
ylim([0 100000])

subplot(3,3,8)
scatter(log(testLesionLoadRef+1),log(testLesionLoad+1))
xlim([2 12])
ylim([2 12])
ylabel('log(x+1) transformed Predictions')
xlabel('log(x+1) transformed Labels')
title('Test lesion load')

subplot(3,3,9)
scatter(testLesionLoadRef,testDice)
xlim([0 100000])
ylim([0 1])
xlabel('Lesion load (from labels)')
ylabel('Dice')
title('Effect of lesion load on Dice - test')

% 
% trainingIntersection = trainingLabels.*uint16(squeeze(trainingPred(:,:,2,:)>inflammThresh));
% trainingUnion = trainingLabels+uint16(squeeze(trainingPred(:,:,2,:)>inflammThresh));
% trainingDice = 2*sum(trainingIntersection,'all') ./ sum(trainingUnion,'all')
% 
% slicewiseTrainingVolume = sum(trainingLabels,[1 2]);
% slicewiseTrainingIntersection = sum(trainingIntersection,[1 2]);
% slicewiseTrainingUnion = sum(trainingUnion,[1 2]);
% slicewiseTrainingDice = 2*slicewiseTrainingIntersection ./ slicewiseTrainingUnion;
% 
% figure
% subplot(1,2,1)
% scatter(slicewiseTrainingVolume(:), slicewiseTrainingDice(:))
% title('Training dataset - VHI vs Dice')
% xlabel('VHI')
% ylabel('Dice')
% ylim([0 1])
% hold on 
% 
% trainingDiceMean = nanmean(slicewiseTrainingDice)
% 
% %8.2 Validation 
% 
% validationIntersection = validationLabels.*uint16(squeeze(validationPred(:,:,2,:)>inflammThresh));
% validationUnion = validationLabels+uint16(squeeze(validationPred(:,:,2,:)>inflammThresh));
% validationDice = 2*sum(validationIntersection,'all') ./ sum(validationUnion,'all')
% 
% slicewiseValidationVolume = sum(validationLabels,[1 2]);
% slicewiseValidationIntersection = sum(validationIntersection,[1 2]);
% slicewiseValidationUnion = sum(validationUnion,[1 2]);
% slicewiseValidationDice = 2*slicewiseValidationIntersection ./ slicewiseValidationUnion;
% 
% subplot(1,2,2)
% scatter(slicewiseValidationVolume(:), slicewiseValidationDice(:))
% title('Validation dataset - VHI vs Dice')
% xlabel('VHI')
% ylabel('Dice')
% ylim([0 1])
% 
% validationDiceMean = nanmean(slicewiseValidationDice)



%% 9. Save parameters and results

save('/scratch0/NOT_BACKED_UP/timbray/Segmentation-Inflammation/Results/resultsFile.mat','results','options','unet','encoderDepth')


