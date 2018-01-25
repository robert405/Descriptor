from NetUtils import *
from Utils import *
import time

sess = tf.Session()

saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, "./model/model.ckpt")

graph = tf.get_default_graph()
cross_entropy = graph.get_tensor_by_name("TrainStep/CrossEntropy:0")
accuracy = graph.get_tensor_by_name("Accuracy/MeanAccuracy:0")


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./result", sess.graph)
tvars = tf.trainable_variables()
saver = tf.train.Saver(tvars)

lr = 1e-5
trainingImgPath = "D:/DataSet/ImageNet2012/NormTrainImg/batch-"

fileSaveName = trainingImgPath + str(641) + ".npy"
testBatchData = np.load(fileSaveName)
testData = testBatchData[0:100,0]
testData = np.stack(testData)
testLabel = testBatchData[0:100,1]
testAccuracy = accuracy.eval(session=sess,feed_dict={"x:0":testData,"y:0":testLabel})
print("Test accuracy : " + str(testAccuracy))

globalStartTime = time.time()

# 641 files, utilise 1-640 pour training, 641 pour test
nbFiles = 640
nbImg = 2000
batchSize = 20
counter = 1
epoch = 1

for k in range(epoch):

    print("==================================================")
    print("Doing epoch no " + str(k))
    print("Learning rate : " + str(lr))
    print("==================================================")

    for i in range(1,nbFiles+1,1):

        localStartTime = time.time()
        fileSaveName = trainingImgPath + str(i) + ".npy"
        batchData = np.load(fileSaveName)
        print("Epoch no " + str(k))
        print("Processing file no " + str(i))

        for j in range(0,nbImg-1,batchSize):

            data = batchData[j:j+batchSize,0]
            label = batchData[j:j+batchSize,1]
            data = np.stack(data)
            data, label = dataAugmentation(data,label)

            if (j == 0):

                summary,trainAccuracy,loss = sess.run([merged,accuracy,cross_entropy],feed_dict={"x:0":data,"y:0":label})
                counter += 1
                writer.add_summary(summary,counter)
                print("Loss : "+str(loss))
                print("Train accuracy : " + str(trainAccuracy))

            sess.run(["TrainStep/Training"],feed_dict={"x:0":data,"y:0":label,"Learning_Rate:0":lr})


        localEndTime = time.time()
        localElapsedTime = localEndTime - localStartTime
        print("Local elapsed time (sec) : " + str(localElapsedTime))
        print("--------------------------------------------------")


globalEndTime = time.time()
globalElapsedTime = globalEndTime - globalStartTime

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
print("Global elapsed time (min): " + str(globalElapsedTime/60))
print("Nb img per file : " + str(nbImg))
print("Nb file done : " + str(nbFiles))
print("Nb epoch done : " + str(epoch))

saver.save(sess, "./model/model.ckpt")



















