from Utils import *

path = "D:\\DataSet\\ImageNet2012\\NormTrainImg\\batch-"
batchSize = 5
fileSaveName = path + str(1) + ".npy"
batchData = np.load(fileSaveName)

for i in range(1,25,5):

    start = i
    imgs = batchData[start:start+batchSize,0]
    imgs = np.stack(imgs)
    labels = batchData[start:start+batchSize,1]
    imgs, labels = dataAugmentation(imgs,labels)

    print(labels)
    showImgs(imgs, 3, batchSize)


"""
513 520 288 407 299 
513 520 288 407 299 
513 520 288 407 299
"""
