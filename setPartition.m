function [datasetInfoMod] =  setPartition(datasetInfo)
% function [datasetInfoMod] =  setPartition(datasetInfo)

%Define number of patients
n=numel(datasetInfo.id)/2;

%Replace id 30 with 29 (missing patient)
datasetInfo.id(datasetInfo.id==30)=29;

%% Get mean SPARCC score for each ID over pre- and post-treatment scans by finding the two SPARCC values for each value of id, such that they are indexed by id
for k=1:n

    meanSparccPrePost(k) = mean(datasetInfo.meanSPARCC(datasetInfo.id==k));
end

%% Sort the mean pre/post SPARCC scores 
[sortedMeans,id] = sort(meanSparccPrePost); %Here ID is the index

%Assign partition labels (1 for training, 2 for validation, 3 for test) to
%give similar inflammation loads to each dataset
partitionScheme=[1 2 3 1 1 1 2 3 1 1 3 2 1 1 1 3 2 1 1 1 1 2 3 1 1 1 2 3 1]';

%% Add partition labels to datasetInfo - 
%To do this, loop through each patient id in the sorted 'id' vector, find
%the corresponding position in the original dataset info vector and update
%the partition label at the same position

for k = 1:n

datasetInfo.partitionLabel(datasetInfo.id == id(k)) = partitionScheme(k);

end

datasetInfo.partitionLabel = datasetInfo.partitionLabel';

figure
subplot(1,3,1)
histogram(datasetInfo.meanSPARCC(datasetInfo.partitionLabel==1),12);
xlim ([0 72])
title('Training distribution')

subplot(1,3,2)
histogram(datasetInfo.meanSPARCC(datasetInfo.partitionLabel==2),12);
title('Validation distribution')
xlim ([0 72])

subplot(1,3,3)
histogram(datasetInfo.meanSPARCC(datasetInfo.partitionLabel==3),12);
title('Test distribution')
xlim ([0 72])

%Export modified version 
datasetInfoMod = datasetInfo;

end
