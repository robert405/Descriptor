from Utils import *

def cmyk_to_rgb(c,m,y,k):
    rgb_scale = 255
    cmyk_scale = 800
    r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    return r,g,b

def convert(cmyk):

    rgb = np.zeros((224,224,3))

    for i in range(224):
        for j in range(224):
            c = cmyk[i,j,0]
            m = cmyk[i,j,1]
            y = cmyk[i,j,2]
            k = cmyk[i,j,3]

            r,g,b = cmyk_to_rgb(c,m,y,k)

            rgb[i,j,0] = r
            rgb[i,j,1] = b
            rgb[i,j,2] = g

    return rgb

path = "D:\\DataSet\\ImageNet2012\\NormTrainImg\\batch-"
#files = [15,136,204,266,267,271,283,284,296,306,325,344,371,373,404,407,502,513,525,591,597,613,631]
files = [266,267,271,283,284,296,306,325,344,371,373,404,407,502,513,525,591,597,613,631]

for i in files:

    fileSaveName = path + str(i) + ".npy"
    batchData = np.load(fileSaveName)

    print("File no : " + str(i))

    for j in range(batchData.shape[0]):

        if (not batchData[j,0].shape == (224,224,3)):
            img = batchData[j,0]
            img = convert(img)
            batchData[j,0] = img
            print("Line : " + str(j) + " is not (224,224,3)! But is : " + str(batchData[j,0].shape))

    np.save(fileSaveName,batchData)
    print("----------------------------------------------")