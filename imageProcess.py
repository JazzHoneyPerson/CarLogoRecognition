# imageProcess.py
# TianYang Jin, Yu Fu
#
# Preprocess images in ./Logo folder
# Output images in ./TrainingSet folder
# Extract and segment the Car Logo part and convert to grayscale
#
# Этот код берет из папки Logo фотки, накладывает на них филбтр и сохраняет все в training set

import glob
import os
from skimage import  io, transform
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.color import rgb2grey

import Base


def getOutputPath(l, logo, num):
    Path = ''
    for i in range(len(l)-1):
        Path += l[i] + '/'
    Path += logo + str(num) + '.jpg'
    return Path


def processOneImage(inputPath, outputPaths):
    image = io.imread(inputPath)#получаем массив пикселей
    greyImage = rgb2grey(image)#переводим в серый цвет
    threshold = threshold_otsu(greyImage)#высчитываем порог вхождения пикселей
    imgout = closing(greyImage > threshold, square(1))#булевый массив, если пиксель проходит порог, то true, иначе false
    imgout = crop(imgout)#обрезаем, убирая части где порог не пройден
    imgout = transform.resize(imgout, (max(imgout.shape), max(imgout.shape)))#переводим булы в цифры
    for outputPath in outputPaths:
        io.imsave(outputPath, imgout)#сохраняем


def crop(a):
    minr = 0
    for r in range(a.shape[0]):
        if all(a[r, :] == 1):
            minr += 1
        else:
            break
    maxr = a.shape[0]-1
    for r in range(a.shape[0]-1, -1, -1):
        if all(a[r, :] == 1):
            maxr -= 1
        else:
            break
    minc = 0
    for c in range(a.shape[1]):
        if all(a[:, c] == 1):
            minc += 1
        else:
            break
    maxc = a.shape[1]-1
    for c in range(a.shape[1]-1, -1, -1):
        if all(a[:, c] == 1):
            maxc -= 1
        else:
            break
    return a[minr:maxr, minc:maxc]


if not os.path.isdir("./TrainingSet/"):
    os.mkdir("./TrainingSet/")


logos = Base.logos#машинки

for logo in logos:
    num = 1
    for image in glob.glob('./Logos/' + logo + '/*.*'):#проходимся по всему списку папки с логотипом
        if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png') or image.endswith('.bmp'):
            inputPath = image
            outputPath1 = getOutputPath(image.split('/'), logo, num)

            outputPath2_dir = "./TrainingSet/" + logo + "/"
            if not os.path.isdir(outputPath2_dir):
                os.mkdir(outputPath2_dir)

            outputPath2 = outputPath2_dir + str(num) + ".jpg"
            num += 1
            try:
                processOneImage(image, [outputPath2])#фильтруем картинку
            except Exception:
                pass
