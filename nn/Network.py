import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys,os


best_acc,node1,node2=0,-1,-1

def sigmoid(x):
    return 1/(1+np.e**-x)

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    a=np.e**x
    b=np.e**-x
    return (a-b)/(a+b)

def onelayer_model(events,power,layer_node,act_func,path,program_name):
    """
    单隐含层模型训练

    :param events: 事件数据
    :param power: 功耗数据
    :param layer_node: 隐含层节点数
    :param act_func: 激活函数
    :param path: 存储路径
    :param program_name: 程序名
    :return:
    """
    tf.reset_default_graph()
    graph=tf.Graph()
    with graph.as_default() as g:
        scalar = MinMaxScaler()
        power_trans = power
        events_trans = scalar.fit_transform(events)
        activate_func = act_func
        data_size = len(events_trans)
        INPUT_FEATURE, LAYER_NODE, OUTPUT_FEATURE = 6, layer_node, 1
        x = tf.placeholder(tf.float32, [None, INPUT_FEATURE], name='x')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_FEATURE], name='y_')

        weight1 = tf.Variable(tf.random_normal([INPUT_FEATURE, LAYER_NODE], stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.0, shape=[LAYER_NODE]))
        weight2 = tf.Variable(tf.random_normal([LAYER_NODE,OUTPUT_FEATURE], stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_FEATURE]))

        layer = activate_func(tf.matmul(x, weight1) + bias1)
        y = tf.matmul(layer, weight2) + bias2
        global_step = tf.Variable(0)
        loss = tf.reduce_mean(tf.square(y_ - y))
        decay_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.98, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(decay_rate).minimize(loss, global_step=global_step)
        batch_size = 40
        print("******Begin Trainning "+program_name+" node:"+str(layer_node)+"******")
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            train_steps = 1000
            for i in range(train_steps):
                start = i * batch_size % data_size
                end = min(start + batch_size, data_size)
                __, w1, b1, w2, b2= sess.run([train_op, weight1, bias1, weight2, bias2],
                                                      feed_dict={x: events_trans[start:end], y_: power_trans[start:end]})
                if i % 100 == 0:
                    loss_d = sess.run(loss, feed_dict={x: events_trans, y_: power_trans})
                    print("After %d training steps,loss on all data is %g" % (i, loss_d))

            l1 = activate_func(tf.matmul(events_trans, weight1) + bias1)
            prediction = sess.run(tf.matmul(l1, weight2) + bias2)
            avg_loss = np.mean(np.abs(prediction - power_trans) / power_trans)
            global best_acc,node1
            if 1-avg_loss>best_acc:
                best_acc=1-avg_loss
                node1=LAYER_NODE
            print("Avg loss is %f" % (avg_loss))
            print("******End Trainning******\n")
            file_name=path+"/onelayer/prediction_acc"
            if not os.path.exists(path+"/onelayer"):
                os.mkdir(path+"/onelayer")
            with open(file_name,"a+",encoding="utf-8") as f:
                dd="onelayer_model "+"node:"+str(layer_node)+" "+str(1-avg_loss)+"\n"
                f.write(dd)

            def model(events):

                xx = scalar.transform(events)
                if act_func==tf.nn.sigmoid:
                    l1 = sigmoid(np.matmul(xx, w1) + b1)
                elif act_func==tf.nn.relu:
                    l1=relu(np.matmul(xx,w1)+b1)
                elif act_func==tf.nn.tanh:
                    l1=tanh(np.matmul(xx,w1)+b1)
                else:
                    print("Activate Function Error!")
                    sys.exit(1)

                l2 = np.matmul(l1, w2) + b2
                return l2

            return model

def twolayer_model(events,power,layer1_node,layer2_node,act_func,path,program_name):
    """
    双隐含层模型

    :param events: 性能事件数据
    :param power: 功耗数据
    :param layer1_node: 第一隐含层节点数
    :param layer2_node: 第二隐含层节点数
    :param act_func: 激活函数
    :param path: 存储路径
    :param program_name: 程序名
    :return:
    """
    tf.reset_default_graph()
    graph=tf.Graph()
    with graph.as_default() as g:
        scalar = MinMaxScaler()
        power_trans = power
        events_trans = scalar.fit_transform(events)
        activate_func=act_func
        data_size = len(events_trans)
        INPUT_FEATURE, LAYER1_NODE, LAYER2_NODE,OUTPUT_FEATURE = 6, layer1_node,layer2_node, 1
        x = tf.placeholder(tf.float32, [None, INPUT_FEATURE], name='x')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_FEATURE], name='y_')

        weight1 = tf.Variable(tf.random_normal([INPUT_FEATURE, LAYER1_NODE], stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]))
        weight2=tf.Variable(tf.random_normal([LAYER1_NODE,LAYER2_NODE],stddev=0.1))
        bias2=tf.Variable(tf.constant(0.0,shape=[LAYER2_NODE]))
        weight3 = tf.Variable(tf.random_normal([LAYER2_NODE, OUTPUT_FEATURE], stddev=0.1))
        bias3 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_FEATURE]))

        layer1 = activate_func(tf.matmul(x, weight1) + bias1)
        layer2= activate_func(tf.matmul(layer1,weight2)+bias2)
        y = tf.matmul(layer2, weight3) + bias3
        global_step = tf.Variable(0)
        loss = tf.reduce_mean(tf.square(y_ - y))
        decay_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.98, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(decay_rate).minimize(loss, global_step=global_step)
        batch_size = 40
        print("******Begin Trainning " + program_name + " node1:" + str(layer1_node) + " node2:"+str(layer2_node)+"******")
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            train_steps = 1000
            for i in range(train_steps):
                start = i * batch_size % data_size
                end = min(start + batch_size, data_size)
                __,w1,b1,w2,b2,w3,b3=sess.run([train_op,weight1,bias1,weight2,bias2,weight3,bias3], feed_dict={x: events_trans[start:end], y_: power_trans[start:end]})
                if i % 100 == 0:
                    loss_d = sess.run(loss, feed_dict={x: events_trans, y_: power_trans})
                    print("After %d training steps,loss on all data is %g" % (i, loss_d))

            l1 = activate_func(tf.matmul(events_trans, weight1) + bias1)
            l2=activate_func(tf.matmul(l1,weight2)+bias2)
            prediction = sess.run(tf.matmul(l2, weight3) + bias3)
            avg_loss = np.mean(np.abs(prediction - power_trans) / power_trans)
            global best_acc,node1,node2
            if 1 - avg_loss > best_acc:
                best_acc = 1 - avg_loss
                node1=LAYER1_NODE
                node2=LAYER2_NODE
            print("Avg loss is %f" % (avg_loss))
            print("******End Trainning******\n")
            file_name=path+"/twolayer/prediction_acc"
            if not os.path.exists(path+"/twolayer"):
                os.mkdir(path+"/twolayer")
            with open(file_name,"a+",encoding="utf-8") as f:
                dd="twolayer_model"+" node1:"+str(layer1_node)+" node2:"+str(layer2_node)+" "+str(1-avg_loss)+"\n"
                f.write(dd)

            def model(events):

                xx=scalar.transform(events)
                if act_func==tf.nn.relu:
                    l1=relu(np.matmul(xx,w1)+b1)
                    l2=relu(np.matmul(l1,w2)+b2)
                elif act_func==tf.nn.sigmoid:
                    l1=sigmoid(np.matmul(xx,w1)+b1)
                    l2=sigmoid(np.matmul(l1,w2)+b2)
                elif act_func==tf.nn.tanh:
                    l1=tanh(np.matmul(xx,w1)+b1)
                    l2=tanh(np.matmul(l1,w2)+b2)
                else:
                    print("Activate Function Error!")
                    sys.exit(1)

                l3=np.matmul(l2,w3)+b3
                return l3

            return model


def linear_model_train(events,power,path,program_name):
    """
    线性模型

    :param events: 事件数据
    :param power: 功耗数据
    :param path: 存储路径
    :param program_name: 程序名
    :return:
    """
    tf.reset_default_graph()
    graph=tf.Graph()
    with graph.as_default() as g:
        scalar = MinMaxScaler()
        power_trans = power
        events_trans = scalar.fit_transform(events)
        data_size = len(events_trans)
        INPUT_FEATURE, OUTPUT_FEATURE = 6,1
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
        print("******Begin Trainning " + program_name + "******")
        with tf.Session(graph=g) as sess:
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
            print("******End Trainning******\n")
            file_name = path + "/linear/prediction_acc"
            if not os.path.exists(path + "/linear"):
                os.mkdir(path + "/linear")
            with open(file_name, "a+", encoding="utf-8") as f:
                dd = "linear_model "+str(1 - avg_loss) + "\n"
                f.write(dd)
            def model(events):
                return np.matmul(scalar.transform(events),w)+b
            return model

def graph(prediction,real,program_name,save_path):
    """
    绘图

    :param prediction: 预测值
    :param real: 真实值
    :param program_name: 程序名
    :param save_path: 存储路径
    :return:
    """
    size=len(real)
    x=np.arange(size)
    plt.figure()
    plt.plot(x,prediction,color='g',label='Prediction')
    plt.plot(x,real,color='r',label='Real')
    plt.title(program_name+" Power Prediction")
    plt.xlabel('Count')
    plt.ylabel('Average Power (W)')
    plt.legend()
    plt.savefig(save_path,dpi=300)
    plt.close()
