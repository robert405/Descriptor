from NetUtils import *
from Utils import *
import time

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name = "x")
x_norm = tf.divide(tf.subtract(x, tf.to_float(128)), tf.to_float(128), name="imgNorm")
y = tf.placeholder(tf.float32, shape=[None], name = "y")

#-------------------------------------------------------------

h_conv = descriptorNet(x_norm)

description = tf.identity(h_conv, name="descriptor")

h_avPool = tf.layers.average_pooling2d(description, [7, 7], [7, 7], name="Avg_Pooling")
inputSize = 1 * 1 * 512
h_avPool_flat = tf.reshape(h_avPool, [-1, inputSize], name = "Flatening")

y_pred = fullyLayerNoRelu("1", h_avPool_flat, 512, 1000)

learning_rate = tf.placeholder(tf.float32, shape=[], name = "Learning_Rate")
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=1000)

with tf.name_scope("TrainStep"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=y_pred), name="CrossEntropy")
    train_step = tf.train.AdamOptimizer(learning_rate, name="Training").minimize(cross_entropy)

tf.summary.scalar('Cross_Entropy',cross_entropy)

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(onehot_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="MeanAccuracy")

tf.summary.scalar('Train_Accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./result", sess.graph)

init = tf.global_variables_initializer()
tvars = tf.trainable_variables()
saver = tf.train.Saver(tvars)
sess.run(init)

lr = 1e-4
trainingImgPath = "D:/DataSet/ImageNet2012/NormTrainImg/batch-"

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

                summary,trainAccuracy,loss = sess.run([merged,accuracy,cross_entropy],feed_dict={x:data,y:label})
                counter += 1
                writer.add_summary(summary,counter)
                print("Loss : "+str(loss))
                print("Train accuracy : " + str(trainAccuracy))

            sess.run([train_step],feed_dict={x:data,y:label,learning_rate:lr})


        localEndTime = time.time()
        localElapsedTime = localEndTime - localStartTime
        print("Local elapsed time (sec) : " + str(localElapsedTime))
        print("--------------------------------------------------")


globalEndTime = time.time()
globalElapsedTime = globalEndTime - globalStartTime

fileSaveName = trainingImgPath + str(641) + ".npy"
testBatchData = np.load(fileSaveName)
testData = testBatchData[0:100,0]
testData = np.stack(testData)
testLabel = testBatchData[0:100,1]
testAccuracy = accuracy.eval(session=sess,feed_dict={x:testData,y:testLabel})

print("Global elapsed time (min): " + str(globalElapsedTime/60))
print("Test accuracy : " + str(testAccuracy))
print("Nb img per file : " + str(nbImg))
print("Nb file done : " + str(nbFiles))
print("Nb epoch done : " + str(epoch))

saver.save(sess, "./model/model.ckpt")



















