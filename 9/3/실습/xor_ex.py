import numpy as np

X = np.array([[0,0],[0,1], [1,0], [1,1]]) # (4, 2)

y = np.array([[0],[1],[1],[0]]) # (4, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# 신경망 클래스
class SimpleNN:
    # 초기화
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        print(f"w1: {self.w1.shape}")
        self.b1 = np.zeros((1, hidden_size))

        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        print(f"w2: {self.w2.shape}")
        self.b2 = np.zeros((1, output_size))

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    # 순전파
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1 # (4, 4)
        # print(f"z1: {self.z1.shape}") 
        self.a1 = sigmoid(self.z1) # (4, 4)
        # print(f"a1: {self.a1.shape}") 

        self.z2 = np.dot(self.a1, self.w2) + self.b2 # (4, 1)
        # print(f"z2: {self.z2.shape}")
        self.a2 = sigmoid(self.z2) # (4, 1)
        # print(f"a2: {self.a2.shape}")

        return self.a2

    # 역전파
    def backward(self, X, y, learning_rate):
        # X : (4, 2)
        m = X.shape[0] # 4

        # l = np.mean((a2 - y) ** 2)


        # dl_dw2 = dl_da2 * da2_dz2 * dz2_dw2
        # dl_dw2 = dl_dz2 * dz2_dw2

        # 2는 의미 없어서 생략
        # 1/m은 dw2에서 가중치의 기울기를 평균화하기 위해 넘긴다.
        dl_da2 = (1/m) * (self.a2 - y)

        da2_dz2 = sigmoid_derivative(self.z2)
        # 따라서
        dl_dz2 = dl_da2 * da2_dz2 # (4, 1)
        dz2_dw2 = self.a1.T 

        # 문제는 여기서 발생합니다!
        # ∂z2/∂w2 = a1이지만, 이는 벡터입니다.
        # 실제로는:
        # a1은 (4, 4) 행렬 (4개 샘플, 4개 은닉 뉴런)
        # w2는 (4, 1) 벡터
        # z2는 (4, 1) 벡터
        # 벡터/행렬에 대한 편미분에서는 연쇄법칙이 다르게 적용됩니다!
        # 올바른 벡터 미분
        # 벡터 z2를 가중치 w2로 미분할 때:
        # ∂z2/∂w2 = a1^T  (전치행렬)

        # 따라서: ∂L/∂w2 = (1/m) * a1^T * ∂L/∂z2
        #              = (1/m) * a1^T * (a2 - y) * sigmoid'(z2)

        # 2. np.dot(self.a1.T, dl_dz2)인 이유
        # 차원을 보면:
        # a1: (4, 4) - 4개 샘플, 4개 은닉 뉴런
        # a1.T: (4, 4) - 전치해도 같은 차원
        # dl_dz2: (4, 1) - 4개 샘플에 대한 오차
        # w2: (4, 1) - 4개 은닉 뉴런에서 1개 출력으로
        # 목표: ∂L/∂w2를 계산해야 하는데, 이는 w2와 같은 차원 (4, 1)이어야 합니다.
        # 행렬 곱셈 규칙:
        # (4, 4) × (4, 1) = (4, 1)  ✓ 올바름 <- np.dot(self.a1.T, d1_dz2)
        # (4, 1) × (4, 4) = 불가능  ✗ 차원 불일치 <- np.dot(dl_dz2, self.a1.T)

        # 따라서 
        # 따라서: dl_dw2 = a1^T @ dl_dz2
        dl_dw2 = np.dot(dz2_dw2, dl_dz2)
        # print(f"dl_dw2: {dl_dw2.shape}")


        # dl_db2 = dl_dz2 * dz2_db2
        dz2_db2 = 1
        # 따라서
        # dl_db2 = dl_dz2

        dl_db2 = np.sum(dl_dz2, axis=0, keepdims=True)
        # print(f"dl_db2: {dl_db2}")
        # b2는 (1, 1) 형태의 2차원 배열입니다!
        # 왜 sum을 사용하는가?
        # 1. 배치 데이터의 특성:
        # dl_dz2: (4, 1) - 4개 샘플, 1개 출력
        # 각 샘플마다 편향 b2에 대한 기울기가 있음
        # 2. 편향의 역할:
        # 편향 b2는 모든 샘플에 공통으로 적용됨
        # 따라서 모든 샘플의 기울기를 합산해야 함
        # 3. 수학적 원리:
        # ∂L/∂b2 = Σ(∂L/∂z2[i])  # i는 샘플 인덱스
        # 예시로 이해하기
        # 4개 샘플이 있다면:        
        # 샘플 1: dl_dz2[0] = 0.1
        # 샘플 2: dl_dz2[1] = -0.2  
        # 샘플 3: dl_dz2[2] = 0.3
        # 샘플 4: dl_dz2[3] = -0.1
        # 편향의 기울기:
        # dl_db2 = 0.1 + (-0.2) + 0.3 + (-0.1) = 0.1


        # dl_dw1 = dl_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
        # dl_dz2 (4, 1)
        dz2_da1 = self.w2.T # (1, 4)
        da1_dz1 = sigmoid_derivative(self.z1) # (4, 4)
        dz1_dw1 = X.T # (2, 4)
        # 따라서 

        dl_da1 = np.dot(dl_dz2, dz2_da1) # (4, 4)
        dl_dz1 = dl_da1 * da1_dz1 # (4, 4)
        dl_dw1 = np.dot(dz1_dw1, dl_dz1) #(2, 4)
        

        # dl_db1 = dl_dz1 * dz1_db1
        dz1_db1 = 1
        dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)

        self.w2 -= learning_rate * dl_dw2
        self.b2 -= learning_rate * dl_db2
        self.w1 -= learning_rate * dl_dw1
        self.b1 -= learning_rate * dl_db1

        return dl_dw1, dl_db1, dl_dw2, dl_db2

    # 학습
    def train(self, X, y, epochs, learning_rate = 0.1):
        losses = []

        for epoch in range(epochs):
            output = self.forward(X)

            loss = np.mean((output - y) ** 2)
            losses.append(loss)

            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"epoch: {epoch}, loss: {loss:.4f}")
        return losses

nn = SimpleNN(2, 4, 1)

losses = nn.train(X, y, 10000, 1.0)

# 학습 후 예측/정확도 출력
proba = nn.forward(X)
pred = (proba >= 0.5).astype(int)
print(f"proba: {np.round(proba.ravel(), 4)}")
print(f"pred : {pred.ravel()}")
print(f"true : {y.ravel()}")
print(f"acc  : {(pred.ravel() == y.ravel()).mean()}")
