from mnist import load_data
import matplotlib.pyplot as plt
from cnn import activate_nn
import numpy as np
from tqdm import tqdm

testIMG_PATH = 'data\\t10k-images.idx3-ubyte'
testLAB_PATH = 'data\\t10k-labels.idx1-ubyte'
trainIMG_PATH = 'data\\train-images.idx3-ubyte'
trainLAB_PATH = 'data\\train-labels.idx1-ubyte'

trainData = load_data(trainIMG_PATH, trainLAB_PATH)
testData = load_data(testIMG_PATH, testLAB_PATH)

def byte_to_int(data_df, img = True):
    arr = []
    
    pbar = tqdm()
    pbar.reset(total=data_df.data.shape[0])

    if img:
        for i in range(data_df.data.shape[0]):
            arr.append(data_df.getIntArray(i))
            pbar.update()
    else:
        for i in range(data_df.data.shape[0]):
            arr.append(data_df.getLabel(i))
            pbar.update()
        

    pbar.refresh()
    pbar.close()
    
    return arr



train_df = [byte_to_int(trainData), byte_to_int(trainData, False)]
test_df = [byte_to_int(testData), byte_to_int(testData, False)]

activate_nn(train_df, test_df)