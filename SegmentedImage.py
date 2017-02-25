from PIL import Image
import glob
import os

# return a cropped image
def getLeftImg(image):

    xsize, ysize = image.size
    newImage = image.crop((0, 0, ysize, ysize))

    return newImage



# return a cropped image
def getRightImg(image):

    xsize, ysize = image.size
    newImage = image.crop((xsize - ysize, 0, xsize, ysize))

    return newImage



# save left cropped images to a directory
def saveLeftImgs(img, name, path):

    for img in imgs:
        img[0].save(path + img[1] + '_L.tif')

    return



# save left cropped images to a directory
def saveRightImgs(img, name, path):

    for img in imgs:
        img[0].save(path + img[1] + '_R.tif')

    return


# main program starts here

image_list = []

for filename in glob.glob('C:/Users/jesmond/Desktop/Kaggle Project/train/*.tif'):
    
    img = Image.open(filename)
    imgName = os.path.splitext(os.path.basename(filename))[0]
    
    leftImg = getLeftImg(img)
    rightImg = getRightImg(img)
    
    leftImg.save('C:/Users/jesmond/Desktop/Kaggle Project/train_L/' + imgName + '_L.tif')
    rightImg.save('C:/Users/jesmond/Desktop/Kaggle Project/train_R/' + imgName + '_R.tif')

    img.close()




