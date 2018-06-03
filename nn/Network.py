import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)

def train():
    mnist = input_data.read_data_sets("MNIST/", one_hot=True)
    INPUT_NODE, OUTPUT_NODE = 784, 10
    LAYER1_NODE = 500
    BATCH_SIZE = 100
    LEARNING_RATE, LEARNING_RATE_DECAY = 0.8, 0.99
    RE_RATE = 0.0001
    TRAINING_STEP = 30000
    MOVING_AVERAGE_DECAY = 0.99
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y_')

    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y=inference(x,None,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    re=tf.contrib.layers.l2_regularizer(RE_RATE)
    regularization=re(weights1)+re(weights2)
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,int(mnist.train.num_examples/BATCH_SIZE),LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEP):
            if i%1000==0:
                validate_acc=sess.run(acc,feed_dict=validate_feed)
                print("After %d training steps,validation accuracy using average model is %g " % (i,validate_acc))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(acc,feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % (TRAINING_STEP,test_acc))


def onelayer_model(events,power):
    scalar=MinMaxScaler()
    power_trans=power
    events_trans=scalar.fit_transform(events)
    data_size=len(events_trans)
    INPUT_FEATURE,LAYER1_NODE,OUTPUT_FEATURE=8,24,1
    x=tf.placeholder(tf.float32,[None,INPUT_FEATURE],name='x')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_FEATURE],name='y_')
    weight1=tf.Variable(tf.random_normal([INPUT_FEATURE,LAYER1_NODE],stddev=0.1))
    bias1=tf.Variable(tf.constant(0.0,shape=[LAYER1_NODE]))
    weight2=tf.Variable(tf.random_normal([LAYER1_NODE,OUTPUT_FEATURE],stddev=0.1))
    bias2=tf.Variable(tf.constant(0.0,shape=[OUTPUT_FEATURE]))

    layer1=tf.nn.sigmoid(tf.matmul(x,weight1)+bias1)
    y=tf.matmul(layer1,weight2)+bias2
    global_step=tf.Variable(0)
    loss=tf.reduce_mean(tf.square(y_-y))
    decay_rate=tf.train.exponential_decay(0.2,global_step,100,0.98,staircase=True)
    train_op=tf.train.AdamOptimizer(decay_rate).minimize(loss,global_step=global_step)
    batch_size=40
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_steps=data_size
        for i in range(train_steps):
            start=i*batch_size%data_size
            end=min(start+batch_size,data_size)
            sess.run(train_op, feed_dict={x: events_trans[start:end], y_: power_trans[start:end]})
            if i%100==0:
                loss_d=sess.run(loss,feed_dict={x:events_trans,y_:power_trans})
                print("After %d training steps,loss on all data is %g" % (i, loss_d))
        l1=tf.nn.sigmoid(tf.matmul(events_trans,weight1)+bias1)
        p=sess.run(tf.matmul(l1,weight2)+bias2)
        avg_loss=np.sum(np.abs(p-power_trans))/np.sum(power_trans)
        print("Data set size :%d\nAverage loss is %.4f" %(data_size,avg_loss))
        x=np.arange(data_size)
        plt.plot(x,p,color='green',label='pred')
        plt.plot(x,power_trans,color='red',label='real')
        plt.xlabel("count")
        plt.ylabel("power /w")
        plt.legend()
        plt.show()


def linear_model_train(events,power):
    scalar = MinMaxScaler()
    power_trans = power
    events_trans = scalar.fit_transform(events)
    data_size = len(events_trans)
    INPUT_FEATURE, OUTPUT_FEATURE = 8,1
    x = tf.placeholder(tf.float32, [None, INPUT_FEATURE], name='x')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_FEATURE], name='y_')
    weight = tf.Variable(tf.random_normal([INPUT_FEATURE, OUTPUT_FEATURE], stddev=0.1))
    bias = tf.Variable(tf.constant(0.0, shape=[OUTPUT_FEATURE]))

    y=tf.matmul(x,weight)+bias
    loss = tf.reduce_mean(tf.square(y_-y))
    global_step=tf.Variable(0)
    decay_rate=tf.train.exponential_decay(0.1,global_step,100,0.98,staircase=True)
    train_op=tf.train.GradientDescentOptimizer(decay_rate).minimize(loss,global_step=global_step)
    batch_size = 40
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_steps = 1000
        for i in range(train_steps):
            start = i * batch_size % data_size
            end = min(start + batch_size, data_size)
            __,w,b=sess.run([train_op,weight,bias], feed_dict={x: events_trans[start:end], y_: power_trans[start:end]})
            if i % 100 == 0:
                loss_d = sess.run(loss, feed_dict={x: events_trans, y_: power_trans})
                print("After %d training steps,loss on all data is %g" % (i, loss_d))

        prediction=sess.run(tf.matmul(events_trans,weight)+bias)
        avg_loss=np.mean(np.abs(prediction-power_trans)/power_trans)
        print("Avg loss is %f" % (avg_loss))
        def model(events):
            return np.matmul(scalar.transform(events),w)+b
        return model

def graph(prediction,real):
    size=len(real)
    x=np.arange(size)
    plt.plot(x,prediction,color='g',label='Prediction')
    plt.plot(x,real,color='r',label='Real')
    plt.xlabel('Count')
    plt.ylabel('Power/W')
    plt.legend()
    plt.show()