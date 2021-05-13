import numpy as np
import random

class N_net():
    def __init__(self):
        self.fitness = 0

        hidden_layer = 48
        self.weight1 = np.random.randn(24, hidden_layer) # 첫번째 가중치 영역 난수배열 생성
        self.weight2 = np.random.randn(hidden_layer, 48) # 두번째 //
        self.weight3 = np.random.randn(hidden_layer, 8) # 세번째 //

    def forward(self, inputs):
        net = np.matmul(inputs, self.weight1) # 인풋배열과 제1 가중치 난수배열 행렬곱
        net = self.relu(net)
        net = np.matmul(net, self.weight2)
        net = self.relu(net)
        net = np.matmul(net, self.weight3)
        net = self.sigmoid(net)
        return net

    def relu(self, x):
        return x * (x >= 0)

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))
"""
출력 레이어 함수 대체
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
"""
"""
class Player():
    player, item = None, None

    def __init__(self, p, N_net):
        self.N_net = N_net
        self.p = p
        self.player = np.array([50,50])
        self.lifetime = 200
        self.gen_item()
        self.num_items = 0
        self.score = 0
        self.fitness = 0
        self.direction = random.randrange(0,8)

    def gen_item(self, group):
        if : # 오브젝트와 겹치지 않는 좌표집합
            self.item = np.array(그 좌표)

    def step(self, direction):
        old_place = self.player[0]

    def get_inputs(self): # 센서입력구하는 함수
        result = np.random.random(24)  # 현재 랜덤인풋값
        return np.array(result)

    def run(self):
        inputs = self.get_inputs()
        outputs = self.N_net.forward(inputs)
        outputs = np.argmax(outputs)
        #print(outputs)

        if outputs == 0: #위로
            self.direction =
        elif outputs == 1: #대각선 오른쪽 위로
            self.direction =
        elif outputs == 2: #오른쪽
            self.direction =
        elif outputs == 3: #대각선 오른쪽 아래로
            self.direction =
        elif outputs == 4: #아래로
            self.direction =
        elif outputs == 5: #대각선 왼쪽 아래로
            self.direction =
        elif outputs == 6: #왼쪽
            self.direction =
        elif outputs == 7: #대각선 왼쪽 위로
            self.direction =
"""
"""
    def action(self, outputs):
        if output
"""
