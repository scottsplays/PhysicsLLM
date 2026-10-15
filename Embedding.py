import mathematics
from tensor import Vector, Matrix
from layer import Layer, Weight

class Embedding:

    def __init__(self, n_tokens, embed_size):
        self.embedding_table = Weight(n_tokens, embed_size, bias = False)

    def __getitem__(self, key):
        return self.embedding_table[key]

    def forward(self, x):
        return Layer(self.embedding_table[x])

    def backward(self, dy, lr):
        return self.embedding_table.backward(dy, lr)

class PositionalEncoding:

    def __init__(self, dim):
        self.dim = dim

    def forward(self):
        temp = []
        for i in range(self.dim):
            if i % 2 == 0:
                temp.append(mathematics.sin(pos / 10000 ** i / self.dim))
            else:
                temp.append(mathematics.cos(pos / 10000 ** i / self.dim))
        return temp

class CrossEntropyLoss:

    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual

        self.loss = None

    def forward(self):
        total = 0
        for p_i, y_i in zip(self.pred, self.actual):
            total -= y_i * mathematics.log(p_i)
        self.loss = total
        return total

    def backward(self):
        temp = []
        for p_i, y_i in zip(self.pred, self.actual):
            temp.append(-1 * y_i / p_i)
        return Layer(temp)

class Softmax:

    def __init__(self, input):
        self.input = input
        self.S = self.forward()

    def forward(self):
        return mathematics.softmax(self.input)

    def backward(self, actual):
        temp = []
        for s_i, y_i in zip(self.S, actual):
            temp.append(s_i * sum(actual) - y_i)
        return Layer(temp)

class LayerNorm:

    def __init__(self, x):
        self.x = x
        self.mu = mathematics.mean(x)
        self.var = mathematics.var(x)

        self.epsilon = 1e-6
        self.gamma = [1] * len(x)
        self.beta = [0] * len(x)

        self.scale = 1 / (self.var + self.epsilon) ** 0.5

        self.normalized = self.forward()

    def forward(self):
        temp = []
        for x, gamma, beta in zip(self.x, self.gamma, self.beta):
            temp.append((x - self.mu) * self.scale) * gamma + beta
        return temp

    def backward(self, lr):
        grad_gamma = [(x - self.mu) * self.scale for x in self.x]
        grad_beta = [1] * len(self.x)

        self.gamma -= lr * grad_gamma
        self.beta -= lr * grad_beta

class SingleHeadAttention:

    def __init__(self, n_embed, head_size):
        self.queries = Linear(n_embed, head_size, bias = False)
        self.keys = Linear(n_embed, head_size, bias = False)
        self.values = Linear(n_embed, head_size, bias = False)



    def forward(self, x):
        scale = (len(x)) ** -0.5
        query = self.queries.forward(x)
        key = self.keys.forward(x)
        value = self.values.forward(x)
        wei = scale * query @ key.transpose()
        return Softmax(wei).forward() @ value

    def backward(self):
        pass

class MultiHeadAttention:

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Linear:

    def __init__(self, n, m, activation_function = mathematics.linear, bias = True):
        self.weights = Weight(n, m, activation_function, bias)

    def forward(self, x):
        return self.weights @ x

    def backward(self, dy, lr):
        return self.weights.backward(dy, lr)

class ReLU:

    def __init__(self):
        self.outputs = None

    def forward(self, vec):
        temp = []
        for x in vec:
            temp.append(mathematics.relu(x))
        self.outputs = Layer(temp)
        return self.outputs

    def backward(self):
        temp = []
        for output in self.outputs:
            temp.append(mathematics.drelu(output))
        return Layer(temp)

class FeedForward:

    def __init__(self, dim):
        self.l1 = Linear(dim, 2 * dim)
        self.l2 = Linear(2 * dim, dim)
        self.ReLU = ReLU()

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.ReLU.forward(x)
        x = self.l2.forward(x)
        return x

    def backward(self, dy, lr):
        dy = self.l2.backward(dy, lr)
        dy = self.ReLU.backward()
        dy = self.l1.backward(dy, lr)

class Block:

    def __init__(self):
        pass

class DecoderOnlyTransformer:

    def __init__(self):
        pass

E = Embedding(65, 32)

S = Softmax([0, 1, 4, 9])
print(S.backward([0, 0, 1, 0]))
