% Demo V1

%Specify number of images to be used from dataset (full training dataset
%n=60,000, test dataset n=10,000)

n=100;

%Import training dataset and labels
[trainImages trainLabels] = readMNIST('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', n, 0);

%Import test dataset and labels
[testImages testLabels] = readMNIST(t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, n, 0);