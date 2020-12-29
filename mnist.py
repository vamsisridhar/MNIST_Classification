import numpy as np

from tqdm import tqdm
#import matplotlib.pyplot as plt
#trainIMG_PATH = 'data\\train-images.idx3-ubyte'
#trainLAB_PATH = 'data\\train-labels.idx1-ubyte'

class ImgData:
    
    # Img Data Structure to store full info about the loaded dataset
    def __init__(self, MGN, TotalImgs, ImgWIDTH, ImgHEIGHT):
        self.MGN = MGN
        self.TotalImgs = TotalImgs
        self.ImgWIDTH = ImgWIDTH
        self.ImgHEIGHT = ImgHEIGHT
        self.quant = TotalImgs * ImgHEIGHT * ImgWIDTH
        self.imgSize = ImgHEIGHT * ImgWIDTH
        self.data = np.empty(int(TotalImgs*ImgWIDTH*ImgHEIGHT), dtype=bytes).reshape(TotalImgs, ImgHEIGHT, ImgWIDTH)
        self.label = np.empty(int(TotalImgs), dtype=bytes)
    

    # Getter Methods for the Class
    def getByteArray(self, n):
        if (n < self.TotalImgs):
            return self.data[n]
        else:
            raise ValueError('index n is out of bounds of the data')
    
    def getIntArray(self, n):
        arr = np.zeros(self.getByteArray(n).shape)
        for i in range(self.getByteArray(n).shape[0]):
            for j in range(self.getByteArray(n).shape[1]):
                arr[i][j] = int.from_bytes(self.getByteArray(n)[i][j], 'big')
        return arr    

    def getLabel(self, n):
        if (n < self.TotalImgs):
            return int.from_bytes(self.label[n], byteorder='big')
        else:
            raise ValueError('index n is out of bounds of the data')


def load_data(PATH_img, PATH_lab):
    # Load Image Path
    with open(PATH_img, 'rb') as f:
        
        # Initialisation Variables iterated
        info = []
        for i in range(4):
            info.append(int.from_bytes(f.read(4), 'big'))
        
        imgContainer = ImgData(*info)

        print(imgContainer.data.shape)


        pbar = tqdm()
        pbar.reset(total = imgContainer.quant)

        # Iterating through data section
        dataptr = 0
        while byte := f.read(1):
            i = dataptr % (imgContainer.imgSize)
            img = dataptr // (imgContainer.imgSize)
            
            x = dataptr % imgContainer.ImgWIDTH
            y = (dataptr // imgContainer.ImgHEIGHT) - (img * imgContainer.ImgHEIGHT)


            imgContainer.data[img, y, x] = byte
            dataptr += 1
            pbar.update()

    pbar.refresh()
    pbar.close()
    f.close()
    # Load Label Path
    with open(PATH_lab, 'rb') as f:
        pbar = tqdm()
        pbar.reset(total = imgContainer.quant)

        #Initialisation Variables
        info = []
        for i in range(2):
            info.append(int.from_bytes(f.read(4), 'big'))
        
        if info[1] == imgContainer.TotalImgs:

            dataptr = 0
            while byte := f.read(1):

                imgContainer.label[dataptr] = byte
                dataptr += 1
                pbar.update()
        else:
            raise ValueError('Label file does not contain sufficient labels')
        
        
    print(imgContainer.label)

    pbar.refresh()
    pbar.close()
    f.close()
    return imgContainer

#testIMG = load_images(testIMG_PATH)
"""
trainIMG = load_data(trainIMG_PATH, trainLAB_PATH)

n = input("Number: ")
print(str(trainIMG.getLabel(int(n))))
plt.imshow(trainIMG.getIntArray(int(n)))
plt.show()
"""