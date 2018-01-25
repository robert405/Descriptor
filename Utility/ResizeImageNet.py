import glob
from Utils import *
from LabelMappingDict import *

path = "./ImageTests/*.JPEG"
allFiles = glob.glob(path)
imgs = []

for file in allFiles:

    img = loadImage(file)
    imgs += [resize(img,128,128)]

showImgs(imgs,1,len(allFiles))

labelMapping = getSynsetToLabelMapping()

print(labelMapping["n01440764"])
print(labelMapping["n01514668"])
