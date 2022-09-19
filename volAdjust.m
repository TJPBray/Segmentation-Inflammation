function adjVol = volAdjust(vol)
% function adjVol = volAdjust(vol)

%Performs imAdjust for each slice in a stack, using limit values derived
%from the whole stack

%Input:
%m x n x p stack of images

%Output:
%m x n x p stack of adjusted images

%Min-max normalise first
vol = vol ./ max(vol,[],'all');

%Get percentile values for whole stack (better to perform for whole stack
%to avoid too much fluctuation)

lower = prctile(vol,1,[1 2 3]);
upper = prctile(vol,99, [1 2 3]);

% %Check results against imadjust for chosen slice
% adjImage1 = imadjust(vol(:,:,12)); %Automatic
% adjImage2 = imadjust(vol(:,:,12),[lower upper]);

%Performs imadjust for each slice of a volume
for k = 1:size(vol,3)
    
    image = vol(:,:,k);
    
    adjImage = imadjust(image,[lower upper])
    
    adjVol(:,:,k) = adjImage;
    
end

end


    
    