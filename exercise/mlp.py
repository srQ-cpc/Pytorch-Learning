import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.001, weight_decay=0.01):
        self.learning_rate = learning_rate
        self. weight_decay =  weight_decay
        self.w1 = np.random.randn(input_size, hidden_size)  # [n, h]
        self.b1 = np.random.randn(1, hidden_size)   # [1, h]
        self.w2 = np.random.randn(hidden_size, 1)   # [h, 1]
        self.b2 = np.random.randn(1, 1)  # [1, 1]
        self.relu = lambda x: np.where(x > 0, x, 0)
        self.d_relu = lambda x: np.where(x > 0, 1, 0)
        self.d1_output = None
        self.a1_output = None
        self.d2_output = None
        self.a2_output = None

    def forward(self, x):
        self.d1_output = np.dot(x, self.w1) + self.b1  # [b, h]
        self.a1_output = self.relu(self.d1_output)
        self.d2_output = np.dot(self.a1_output, self.w2) + self.b2  # [b, 1]
        self.a2_output = self.relu(self.d2_output)
        return self.a2_output

    def backward(self, x, y, y_pred):
        batch_size = x.shape[0]
        err = y_pred - y  # [b, 1]
        loss = np.mean(err * err)
        print(f"loss: {loss}")
        d_a2_output = 2 * err
        d_d2_output = d_a2_output * self.d_relu(self.d2_output)  # [b, 1]
        d_w2 = np.dot(self.a1_output.T, d_d2_output)  # [h, 1]
        d_b2 = np.sum(d_d2_output, axis=0, keepdims=True)   # [1, 1]
        d_a1_output = np.dot(d_d2_output, self.w2.T)  # [b, h]
        d_d1_output = d_a1_output * self.d_relu(self.d1_output)
        d_w1 = np.dot(x.T, d_d1_output)  # [n, h]
        d_b1 = np.sum(d_d1_output, axis=0, keepdims=True)   # [1, h]

        self.w1 -= self.learning_rate * (d_w1 + self.weight_decay * self.w1) / batch_size
        self.b1 -= self.learning_rate * (d_b1 + self.weight_decay * self.b1) / batch_size
        self.w2 -= self.learning_rate * (d_w2 + self.weight_decay * self.w2) / batch_size
        self.b2 -= self.learning_rate * (d_b2 + self.weight_decay * self.b2) / batch_size

    def fit(self, x, y):
        assert y.shape[0] == y.shape[0]
        y_pred = self.forward(x)
        self.backward(x, y, y_pred)


input_size = 4
batch_size = 16

train_x = np.random.rand(128, 4) * 5
train_y = np.sum(train_x, axis=1).reshape(-1, 1)

print(train_x)
print(train_y)

mlp = MLP(input_size, hidden_size=16)

for i in range(10000):
    idx = i % (128 // batch_size) * batch_size
    batch_x = train_x[idx: (idx + batch_size)]
    bacth_y = train_y[idx: (idx + batch_size)]
    mlp.fit(batch_x, bacth_y)


test_x = [[1, 2, 3, 4]]
print(mlp.forward(test_x))