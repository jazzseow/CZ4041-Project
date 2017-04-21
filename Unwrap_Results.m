%% Unwrap into Kaggle-readable format, img(420x580)
% xy1,x1.length(y),xy2,x2.length(y),...
clear all; load('test_masks.mat');

for k = 1:size(test_masks,1)       
    coor = cell2mat(test_masks{k,2});
    img(k) = str2num(cell2mat(test_masks{k,1}));
    
    if(coor == 0)
        pixels(k) = "";
        disp(k);
        continue;
    end
    
    x = abs(coor(1));
    y = abs(coor(2));
    width = coor(3);
    height = coor(4);

    pix = x*420+y;
    pix_str = strcat(string(pix),{' '},string(height));

    for j = 1:width
        pix = pix+420; %next column wraps around at exactly 420 pixels
        pix_str = strcat(pix_str,{' '},string(pix),{' '},string(height));
    end
    
    pixels(k,1) = pix_str;
end

img = img';
% pixels = pixels';

fname = 'test_results';
xlswrite(fname,img);
xlswrite(fname,pixels,'B1:B5508');