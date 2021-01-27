from pandas import read_csv
import networkx as nx
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

from module.GCN import MYGCN
from Hyper_parameter import args

# 随机种子
import os
os.environ['PYTHONHASHSEED']=str(args.seed)

import random
random.seed(args.seed)

import numpy as np
np.random.seed(args.seed)

import tensorflow as tf
tf.random.set_seed(args.seed)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# 数据读取 按类别比例划分 训练集 验证集
attributes = read_csv(args.attribute_path)

x_data = attributes.iloc[:, 1]
y_data = attributes.iloc[:, 2]
print('total positive rate: ', sum(y_data)/len(y_data))

pos_num = sum(y_data)
neg_num = len(y_data) - pos_num
pos_train_num = int(pos_num * args.train_rate)
neg_train_num = int(neg_num * args.train_rate)

pos_data_x = []
neg_data_x = []
pos_data_y = []
neg_data_y = []

for x, y in zip(x_data, y_data):
    if y == 1:
        pos_data_x.append(x)
        pos_data_y.append(y)
    else:
        neg_data_x.append(x)
        neg_data_y.append(y)

x_train = pos_data_x[:pos_train_num]
x_train.extend(neg_data_x[:neg_train_num])
x_train = np.array(x_train)
x_train = x_train.flatten().astype(('float32'))

y_train = pos_data_y[:pos_train_num]
y_train.extend(neg_data_y[:neg_train_num])
y_train = np.array(y_train)
y_train = y_train.flatten().astype('float32')

x_test = pos_data_x[pos_train_num:]
x_test.extend(neg_data_x[neg_train_num:])
x_test = np.array(x_test)
x_test = x_test.flatten().astype(('float32'))

y_test = pos_data_y[pos_train_num:]
y_test.extend(neg_data_y[neg_train_num:])
y_test = np.array(y_test)
y_test = y_test.flatten().astype('float32')


# graph 读取生成邻接矩阵
G = nx.Graph()
path = args.edge_path
edge_list = []
node_set = set()
with open(path, 'r') as f:
    for line in f:
        cols = line.strip().split(',')
        y1=int(cols[0])
        y2=int(cols[1])
        node_set.add(y1)
        node_set.add(y2)
        edge = (y1, y2)
        edge_list.append(edge)

G.add_nodes_from(range(0, len(node_set)))
G.add_edges_from(edge_list)
A = nx.adjacency_matrix(G).todense()
A = np.array(A)

emb_list = []
out_list =[]
epoch_num = args.epoch_num


# input data 处理
train_x = []
train_y = []
for i in range(len(x_train)):
    train_x.append(to_categorical(x_train[i], num_classes=args.X_shape))
    train_y.append(y_train[i])
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
print('positive rate: ',y_train.sum()/len(y_train))
train_x = np.expand_dims(train_x, -1)

valid_x = []
valid_y = []
for i in range(len(x_test)):
    valid_x.append(to_categorical(x_test[i], num_classes=args.X_shape))
    valid_y.append(y_test[i])
valid_x = np.asarray(valid_x)
valid_y = np.asarray(valid_y)
print('negative rate: ', 1-(y_test.sum() / len(y_test)))
valid_x = np.expand_dims(valid_x, -1)


# 模型训练
my_gcn = MYGCN(A, args.X_shape, [args.hidden_dim_1, args.hidden_dim_2])
my_gcn.build(input_shape=[args.X_shape, 1])
tb = TensorBoard()

my_gcn.compile(optimizer=Adam(lr=args.lr), loss=BinaryCrossentropy(), metrics='accuracy')
hist = my_gcn.fit(x=train_x, y=train_y, batch_size=100, validation_data=(valid_x, valid_y), shuffle=True, validation_freq=1, epochs=epoch_num, callbacks=[tb])


# plot train process

import matplotlib.pyplot as plt

def plot_train(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'valid'], loc='upper left')
    plt.savefig(args.logs_dir + "loss_{}.png".format(os.getpid()))
    plt.close()

plot_train(hist)