close all;
clear all;
clc;

%% Load MNIST dataset
load ('mnist.mat');

% Specifiy image to display
id = 18;

% Display the digit
image (rescale(training.images(:,:,id),0,255));

% Scale axis
axis square equal;

% Grayscale
colormap(gray)
colorbar

% Set title
title (sprintf ("Digit: %d", training.labels(id)))