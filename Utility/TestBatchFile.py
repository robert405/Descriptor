from Utils import *
from LabelMappingDict import *

#labelMapping = getSynsetToLabelMapping()
#mappingArray

"""
Full shape : (2000, 2)
One line shape : (2,)
Img shape : (224, 224, 3)
label : 672
"""
inputPath = "D:\\DataSet\\ImageNet2012\\TrainImg"
outputPath = "D:\\DataSet\\ImageNet2012\\NormTrainImg\\batch-"

fileSaveName = outputPath + str(1) + ".npy"
batchData = np.load(fileSaveName)


imgs = batchData[0:5,0]
labels = batchData[0:5,1]
length = imgs.shape[0]


synsets = []
for label in labels:
    synsets += [mappingArray[label]]

print(synsets)

print("Full shape : " + str(batchData.shape))
print("One line shape : " + str(batchData[0].shape))
print("Img shape : " + str(batchData[0,0].shape))
print("label : " + str(batchData[0,1]))

print(str(batchData[0].shape == (2,)))
print(str(batchData[0,0].shape == (224,224,3)))

showImgs(imgs,1,length)
