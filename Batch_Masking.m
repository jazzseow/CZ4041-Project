close all; clear all;

%% Gather all images in folder
images = dir('*tif');

i = 1;
j = 1;

fprintf('There are %d images in total.\n',length(images));

for k = 1:length(images)
    if regexp(images(k).name,'.src.','once')
        srcImages(i) = images(k); %store source images
        i = i + 1;
    elseif regexp(images(k).name,'.mask.','once')
        maskImages(j) = images(k); %store mask images
        j = j + 1;
    end
end
        
fprintf('There are %d source images in total.\n',length(srcImages));

count = 1;

%% Process all images in folder
for k = 1: length(srcImages)
    %temp_mask = imread(maskImages(k).name);
    temp_src = imread(srcImages(k).name);
    
    % First check that nerve exists in image via mask values
%     if sum(sum(temp_mask)) == 0
%         continue; % Skip if no nerve found
%     end
    
    % Generate ground truth bounding box coordinates
    name{count} = [srcImages(k).name(1:end-8) '_proc.tif'];
    maskBBox{count} = ExtractMaskCoordinates(temp_mask);
    count = count+1;
    
    temp_h = histeq(temp_src); temp_sh = imsharpen(temp_h);
    
    % intersect image with mask
    %temp_src(temp_mask == 0) = 0;
    
    % save processed image
    new_name = [srcImages(k).name(1:end-8) '_proc.tif'];
    imwrite(temp_sh, new_name);
end

name = name';
nerve = maskBBox';
nerve_id = table(name, nerve);
save('nerve_id','nerve_id');

%% Check correctnes of ground truth
bbox = cell2mat(nerve_id{2,2});
rgbImage=imread('10_104_mask.tif');
imshow(rgbImage)
hold on
rectangle('Position', bbox,...
          'EdgeColor','r',...
          'LineWidth', 3);

%% Generate coordinates for ground truth
function coor = ExtractMaskCoordinates(maskImg)
    x_min = size(maskImg,1); %start with max height
    y_min = length(maskImg); %start with max length
    height = 0;
    width = 0;
    
    for j = 1:length(maskImg) %horizontal
        for k = 1:size(maskImg,1) %vertical
            if maskImg(k,j) > 0
                x_min = min(x_min,j);
                y_min = min(y_min,k);
                width = max(width,abs(x_min - j));
                height = max(height,abs(y_min - k));
            end
        end
    end
    
    coor = [x_min, y_min, width, height];
end