import numpy as np
import random

class N_net():
    def __init__(self,input,hidden1,hidden2,output):
        self.fitness = 0

        self.input_layer = input
        self.hidden_layer = hidden1   # 48
        self.hidden_layer2 = hidden2   # 48
        self.output_layer = output

        self.weight1 = np.random.randn(self.input_layer, self.hidden_layer) / np.sqrt(self.input_layer / 2)  # He 초기화, 첫번째 가중치 영역 난수배열 생성
        self.weight2 = np.random.randn(self.hidden_layer, self.hidden_layer2) / np.sqrt(self.hidden_layer / 2)  # He 초기화, 두번째 //
        self.weight3 = np.random.randn(self.hidden_layer2, self.output_layer) / np.sqrt(self.hidden_layer2 / 2)  #

    def forward(self, inputs):
        net = np.dot(inputs, self.weight1) # 인풋배열과 제1 가중치 난수배열 행렬곱
        net = self.relu(net)
        net = np.dot(net, self.weight2)
        net = self.relu(net)
        net = np.dot(net, self.weight3)
        net = self.softmax(net)
        return net

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def leaky_relu(self, x):
        return np.maximum(0.1, x)

    def softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c) # 오버플로우 대책
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y
