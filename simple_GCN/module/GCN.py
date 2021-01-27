import tensorflow as tf
import keras
import numpy as np


class GcnLayer(keras.layers.Layer):
    def __init__(self,  A, in_units, out_units, activation=keras.activations.relu):
        super(GcnLayer, self).__init__()
        I = np.eye(*A.shape)
        A_hat = A.copy() + I
        self.in_units = in_units
        self.out_units = out_units
        self.act = activation
        D = np.sum(A_hat, axis=0)
        D_var = tf.Variable(D)
        D_inv = D_var**-0.5
        D_inv = np.diag(D_inv)
        D_inv_var =tf.Variable(D_inv)
        self.A_hat_1 = D_inv_var * A_hat * D_inv_var
        self.A_hat_2=np.array(self.A_hat_1).astype("float32")
        self.A_hat_var = tf.Variable(self.A_hat_2)
        self._fc = keras.layers.Dense(units=self.out_units,input_dim= self.in_units,activation=self.act, kernel_regularizer=keras.regularizers.L2(1e-4))

    def call(self, x):
        x = tf.matmul(self.A_hat_var,x)
        x = self._fc(x)
        x = keras.layers.Dropout(0.5)(x)
        return x


class GCN(keras.layers.Layer):
    def __init__(self,A,input_dim,hidden_dim_list,activation = keras.activations.sigmoid,):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.A = A
        self.hidden_dim_list = hidden_dim_list#[4,2]
        self.gcn1 = GcnLayer(self.A,self.input_dim,self.hidden_dim_list[0]) #34，4
        self.gcn2 = GcnLayer(self.A,self.hidden_dim_list[0],self.hidden_dim_list[1]) #34，2
        self._fc = keras.layers.Dense(units=1,input_dim=self.hidden_dim_list[1],activation=activation, kernel_regularizer=keras.regularizers.L2(1e-5))

    def call(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self._fc(x)
        x = keras.layers.Dropout(0.1)(x)
        return x


class MYGCN(keras.Model):
    def __init__(self,A,input_dim,hidden_dim_list,activation = keras.activations.sigmoid,):
        super(MYGCN, self).__init__()
        self.input_dim = input_dim
        self.A = A
        self.hidden_dim_list = hidden_dim_list#[4,2]
        self.gcn1 = GcnLayer(self.A,self.input_dim,self.hidden_dim_list[0]) #34，4
        self.gcn2 = GcnLayer(self.A,self.hidden_dim_list[0],self.hidden_dim_list[1]) #34，2
        self._fc = keras.layers.Dense(units=1,input_dim=self.hidden_dim_list[1],activation=activation, kernel_regularizer=keras.regularizers.L2(1e-4))

    def call(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self._fc(x)
        x = keras.layers.Dropout(0.2)(x)
        return x



