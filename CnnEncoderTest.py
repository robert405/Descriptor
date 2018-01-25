import tensorflow as tf
from scipy.misc import imread, imresize
from Utility.imagenet_classes import class_names
import numpy as np

sess = tf.Session()

saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, "./model/model.ckpt")

graph = tf.get_default_graph()
prediction = graph.get_tensor_by_name("FullyNoRelu-1/add:0")
descriptor = graph.get_tensor_by_name("descriptor:0")
accuracy = graph.get_tensor_by_name("Accuracy/MeanAccuracy:0")
softmax = tf.nn.softmax(prediction)
print("Model Loaded")

img1 = imread('fish.jpg', mode='RGB')
img1 = imresize(img1, (224, 224))

feed_dict = {"x:0": [img1]}
result = sess.run(softmax, feed_dict)

print(result.shape)

pred = result[0]

print(pred.shape)

onePred = np.argmax(pred)
print(onePred)
print("")

multiPred = pred.argsort()[-5:][::-1]

print("Top 5 Prediction")
for prediction in multiPred:
    print(class_names[prediction])


result = sess.run(descriptor, feed_dict)
description = result[0]

print(description.shape)

trainingImgPath = "D:/DataSet/ImageNet2012/NormTrainImg/batch-"
fileSaveName = trainingImgPath + str(641) + ".npy"
testBatchData = np.load(fileSaveName)
mean = 0
batchsize = 100
nbLoop = int(testBatchData.shape[0] / batchsize)
totalTestImgs = nbLoop*batchsize

for i in range(0,totalTestImgs,batchsize):

    testData = testBatchData[i:i+batchsize,0]
    testData = np.stack(testData)
    testLabel = testBatchData[i:i+batchsize,1]
    testAccuracy = accuracy.eval(session=sess,feed_dict={"x:0":testData,"y:0":testLabel})
    mean += testAccuracy


mean = mean / nbLoop
print("Test accuracy for " + str(totalTestImgs) + " images : " + str(mean))

testData = testBatchData[0:100,0]
testData = np.stack(testData)
testLabel = testBatchData[0:100,1]
testAccuracy = accuracy.eval(session=sess,feed_dict={"x:0":testData,"y:0":testLabel})
print("Test accuracy : " + str(testAccuracy))










