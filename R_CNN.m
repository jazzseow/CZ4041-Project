%% RCC Testbed

load('rcnnStopSigns.mat', 'stopSigns', 'layers');
load('nerve_id.mat');

% imDir = fullfile(matlabroot, 'toolbox', 'vision', 'visiondata',...
%   'stopSignImages');
% addpath(imDir);

options = trainingOptions('sgdm', ...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 10);

rcnn = trainRCNNObjectDetector(nerve_id, layers, options, ...
                               'NegativeOverlapRange', [0 0.3]);

%% Test R-CNN detector on test image
img = imread('1_4_proc.tif');

[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);

%% Display strongest detection result
if(max(score)>0.9)
    disp('Nerve found.');
    [score, idx] = max(score);

    bbox = bbox(idx, :);
    annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

    detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

    figure
    imshow(detectedImg)
else
    disp('No nerve.');
end