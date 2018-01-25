import os
from random import shuffle
import re
from Utils import *
from Utility.LabelMappingDict import *
import time

labelMapping = getSynsetToLabelMapping()
inputPath = "D:\\DataSet\\ImageNet2012\\TrainImg"
outputPath = "D:\\DataSet\\ImageNet2012\\NormTrainImg\\batch-"

# Remember some image have 4 channel known as cmyk

print("Start finding files")

allImages = []

for root, dirs, files in os.walk(inputPath):

    for file in files:
        if (file.endswith(".JPEG")):
            allImages += [os.path.join(root, file)]

shuffle(allImages)

print("Saving in batch")
batchSize = 2000
fileCounter = 0
batchCounter = 0
batch = []
regex = re.compile("n[0-9]+")
startTime = time.time()

for imgPath in allImages:

    synset = re.search(regex,imgPath).group(0)
    img = loadImage(imgPath)
    label = labelMapping[synset]
    img = resize(img,224,224)
    batch += [np.array([img,label])]
    fileCounter += 1

    if (fileCounter % batchSize == 0):
        batchCounter += 1
        print("Saving batch no : " + str(batchCounter))
        batchToSave = np.array(batch)
        fileSaveName = outputPath + str(batchCounter)
        np.save(fileSaveName, batchToSave)
        batch = []

        endTime = time.time()
        elapsedTime = endTime-startTime
        print("Elapsed time (sec) : " + str(elapsedTime))
        print("--------------------------------------------------")
        startTime = time.time()


if (0 < fileCounter - (batchCounter * batchSize)):
    batchCounter += 1
    print("Saving batch no : "+str(batchCounter))
    batchToSave = np.array(batch)
    fileSaveName = outputPath+str(batchCounter)
    np.save(fileSaveName, batchToSave)

print("Nb of file processed : " + str(fileCounter))
print("Nb of batch created : " + str(batchCounter))



