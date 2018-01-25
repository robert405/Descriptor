from Utils import *

"""
Full shape : (2000, 2)
One line shape : (2,)
Img shape : (224, 224, 3)
label : 672

File no : 641
File size is : 1167
"""

path = "D:\\DataSet\\ImageNet2012\\NormTrainImg\\batch-"


for i in range(1,641+1,1):

    fileSaveName = path + str(i) + ".npy"
    batchData = np.load(fileSaveName)
    allTrue = True

    print("File no : " + str(i))
    print("File size is : " + str(batchData.shape[0]))

    for j in range(batchData.shape[0]):

        allTrue = allTrue and batchData[j].shape == (2,)
        allTrue = allTrue and batchData[j,0].shape == (224,224,3)

    if (not allTrue):
        print("File no " + str(i) + " is WRONG!")

    print("----------------------------------------------")