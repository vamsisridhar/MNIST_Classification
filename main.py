from mnist import load_data
import matplotlib.pyplot as plt


testIMG_PATH = 'data\\t10k-images.idx3-ubyte'
testLAB_PATH = 'data\\t10k-labels.idx1-ubyte'
trainIMG_PATH = 'data\\train-images.idx3-ubyte'
trainLAB_PATH = 'data\\train-labels.idx1-ubyte'

trainData = load_data(trainIMG_PATH, trainLAB_PATH)
plt.imshow(trainData.getIntArray(56))
print(trainData.getLabel(56))
plt.show()
