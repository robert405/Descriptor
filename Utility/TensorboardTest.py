from NetUtils import *
import time

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name = "x")
y = tf.placeholder(tf.float32, shape=[None], name = "y")

#-------------------------------------------------------------

h_conv = descriptorNet(x)

h_avPool = tf.layers.average_pooling2d(h_conv, [7, 7], [7, 7])
inputSize = 1 * 1 * 512
h_avPool_flat = tf.reshape(h_avPool, [-1, inputSize], name = "flatening")

y_pred = fullyLayerNoRelu("1", h_avPool_flat, 512, 1000)

learning_rate = tf.placeholder(tf.float32, shape=[], name = "learning_rate")
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=1000)

with tf.name_scope("trainStep"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=y_pred))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(onehot_labels,1))

    with tf.name_scope("train"):
        trainAccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("test"):
        testAccuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.summary.scalar('train_accuracy',trainAccuracy)
tf.summary.scalar('test_accuracy',testAccuracy)

savePath = "C:/Users/Martin/Documents/Free Dev/ProjectResult"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(savePath, sess.graph)
init = tf.global_variables_initializer()
tvars = tf.trainable_variables()
saver = tf.train.Saver(tvars)
sess.run(init)

lr = 1e-4
trainingImgPath = "D:/DataSet/ImageNet2012/NormTrainImg/batch-"

fileSaveName = trainingImgPath + str(641) + ".npy"
testBatchData = np.load(fileSaveName)
testData = testBatchData[0:50,0]
testData = np.stack(testData)
testData = (testData - 128) / 128
testLabel = testBatchData[0:50,1]

globalStartTime = time.time()

# 641 files total, ending training at 640
nbFiles = 3
nbImg = 2000
batchSize = 25
counter = 1
testCounter = 1
epoch = 1
for k in range(epoch):

    print("==================================================")
    print("Doing epoch no " + str(k))
    print("Learning rate : " + str(lr))
    print("==================================================")

    for i in range(1,nbFiles+1,1):

        fileSaveName = trainingImgPath + str(i) + ".npy"
        batchData = np.load(fileSaveName)
        print("Processing file no " + str(i))

        localStartTime = time.time()

        for j in range(0,nbImg-1,batchSize):

            data = batchData[j:j+batchSize,0]
            data = np.stack(data)
            data = (data - 128) / 128
            label = batchData[j:j+batchSize,1]

            if (j % 500 == 0):
                print("Batch step no " + str(j))
                summary,accuracy,_ = sess.run([merged,trainAccuracy,train_step],feed_dict={x:data,y:label,learning_rate:lr})
                counter += 1
                writer.add_summary(summary,counter)
                print("Train accuracy : "+str(accuracy))

            else:
                sess.run([train_step],feed_dict={x:data,y:label,learning_rate:lr})

        # mesure test accuracy and calculate time for processing a file
        summary,accuracy = sess.run([merged,testAccuracy],feed_dict={x:testData,y:testLabel})
        writer.add_summary(summary,counter)
        print("Test accuracy : " + str(accuracy))
        localEndTime = time.time()
        localElapsedTime = localEndTime - localStartTime
        print("Local elapsed time (sec) : " + str(localElapsedTime))
        print("--------------------------------------------------")



testData = testBatchData[50:100,0]
testData = np.stack(testData)
testData = (testData-128)/128
testLabel = testBatchData[50:100,1]

# mesure test accuracy and calculate time for processing all files
globalEndTime = time.time()
globalElapsedTime = globalEndTime - globalStartTime
print("Global elapsed time (min): " + str(globalElapsedTime/60))
summary,accuracy = sess.run([merged,testAccuracy],feed_dict={x:testData,y:testLabel})
writer.add_summary(summary,counter)
print("Test accuracy : " + str(accuracy))
print("Nb img per file : " + str(nbImg))
print("Nb file done : " + str(nbFiles))
print("Nb epoch done : " + str(epoch))

saver.save(sess, savePath + "/model/model.ckpt")





















