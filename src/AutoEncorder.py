# -*- coding: utf-8 -*-
import numpy as np



class Activation:
    """
    活性化関数をまとめたクラス
    """

    def sigmoid(flag,x):
        """
        シグモイド関数
        :param flag: True, 通常 False, 微分後
        :param x: 入力データ
        """
        if flag == True:
            data = 1.0 / (1+np.exp(-x))
            return data
        else:
            return x * (1. - x)

    def relu(flag,x):
        """
        Relu関数
        :param flag: True, 通常 False, 微分後
        :param x: 入力データ
        """
        if flag == True:
            return x * (x > 0)
        else:
            return 1. * (x>0)

    def tanh(flag,x):
        """
        tanh関数
        :param flag: True, 通常 False, 微分後
        :param x: 入力データ
        """
        if flag == True:
            return np.tanh(x)
        else:
            return 1. - x*x



class LossFunction:
    """
    損失関数のクラス
    """
    def mse(flag,x,y):
        """
        MSE関数
        :param flag: True, 通常 False, 微分後
        :param x: 入力データ
        """
        if flag == True:
            return (y-x)*(y-x)/2
        else:
            return y-x




class AutoEncoder:
    """
    自己符号化クラス
    """
    def __init__(self,input_num,hidden_num,activation=Activation.sigmoid):
        """
        コンストラクタ
        :param input_num:   入力次元数
        :param hidden_num:  隠れ層の数
        :param activation:  活性化関数
        """
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.hidden_weight = np.random.randn(hidden_num,input_num+1)
        self.output_weight = np.random.randn(input_num,hidden_num+1)
        self.act = activation

    def _encode(self,data):
        """
        符号化関数
        :param data:    入力データ
        :return :   隠れ層の結果, 計算結果
        """
        h,o = self.__forward(data)
        return h,o

    def __forward(self,data):
        """
        forward計算関数
        :param data:    入力データ
        :return : 隠れ層の結果,計算結果
        """
        # add bias
        input = np.hstack((data,1))
        # calc hidden_value
        hidden_value = self.act(True,self.hidden_weight.dot(input))
        # calc output
        output_value = self.act(True,self.output_weight.dot(np.hstack((hidden_value,1))))
        return hidden_value, output_value

    def __backprop(self,x,hidden,output):
        """
        誤差逆伝播計算
        :param x:       入力データ
        :param hidden:  隠れ層の値
        :param output:  出力値
        """
        # kari : diff
        input = x.reshape(len(x),1)
        output_error = output - x

        #update output_weight
        output_diff = self.act(False,output)
        output_delta = output_diff * output_error
        self.output_weight -= self._learn_rate * output_delta.reshape(-1,1) * np.hstack((hidden,1))
        #update hidden_weight
        hidden_delta = self.act(False,hidden).reshape(-1,1) * np.dot(np.transpose(self.output_weight[:,1:]),output_delta.reshape(-1,1))
        self.hidden_weight -= self._learn_rate * np.transpose(np.transpose(hidden_delta) * np.hstack((x,1)).reshape(-1,1))

    def _fitting(self,data,learn_rate=0.001,epoch=100000):
        """
        フィッティング関数
        :param data:        入力データ
        :param learn_rate:  学習率
        :param epoch:       更新回数
        """
        i = 0
        self._learn_rate = learn_rate
        while i < epoch:
            err = 0.0
            for x in data:
                h,o = self.__forward(x)
                self.__backprop(x,h,o)
                i+=1

            err = self._error(data)
            print(err)

       # h,o = self.__forward(data)

    def _error(self,data):
        """
        エラー計算
        :param data:    入力値
        :return     エラー値
        """
        err = 0.0
        for x in data:
            h,o = self.__forward(x)
            err += (x-o).dot(x-o)/2
        return err
