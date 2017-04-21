%% Test R-CNN detector on test image
load('rcnn_model.mat');

imgList = dir('*proc*');

for k = 1:length(imgList)
    img = imread(imgList(k).name);
    name = imgList(k).name(1:end-9);
    disp(name);
    [bbox, score, label] = detect(rcnn, img,...
                                  'MiniBatchSize', 32,...
                                  'ExecutionEnvironment','gpu');
    
    %% Display strongest detection result
    if(max(score)>0.9)
        % nerve found
        disp('Nerve found.');
        [score, idx] = max(score);
        bbox = bbox(idx, :);
    else
        disp('No nerve.');
        bbox = 0;
    end
    
    nameList{k} = name;
    bbList{k} = bbox;
end

img = nameList';
pixels = bbList';
test_masks = table(img, pixels);
save('test_masks','test_masks');