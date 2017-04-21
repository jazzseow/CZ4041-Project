close all; clear all;

%% Gather all images in folder
images = dir('*tif');

i = 1;
j = 1;

fprintf('There are %d images in total.\n',length(images));

for k = 1:length(images)
    if isempty(regexp(images(k).name,'.proc.','once'))
        srcImages(i) = images(k); %store source images
        i = i + 1;
    end
end
        
fprintf('There are %d source images in total.\n',length(srcImages));

count = 1;

%% Process all images in folder
for k = 1: length(srcImages)
    temp_src = imread(srcImages(k).name);
    
    % process image
    temp_h = histeq(temp_src); temp_sh = imsharpen(temp_h);
    
    % save processed image
    new_name = [srcImages(k).name(1:end-4) '_proc.tif'];
    imwrite(temp_sh, new_name);
end