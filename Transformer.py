import mathematics
from tensor import Vector, Matrix, concatenate
from layer import Layer, Weight

class Embedding:

    def __init__(self, vocab_size, embed_size):
        self.input = None
        self.output = None

        self.embedding_table = Matrix(vocab_size, embed_size)

    def forward(self, input):
        self.input = list(input)

        temp = []
        for key in self.input:
            temp.append(self.embedding_table[key])

        self.output = Matrix(temp)
        return self.output

    def backward(self, dy, lr):
        for i, token in enumerate(self.input):
            row = self.embedding_table[token].to_list()
            self.embedding_table.vectors[token] = Vector([row[j] - lr * dy[i][j] for j in range(len(row))])

class PositionalEncoding:

    def __init__(self, n_tokens, embed_size):
        self.n_tokens = n_tokens
        self.embed_size = embed_size

        self.pos_emb = Embedding(n_tokens, embed_size)

    def forward(self, x):
        n = len(x)
        return self.pos_emb.forward(range(n))

    def backward(self, dy, lr):
        return self.pos_emb.backward(dy, lr)

class CrossEntropyLoss:

    def __init__(self):
        self.pred = None
        self.output = None
        
        self.loss = None

    def forward(self, pred, output):
        self.pred = pred
        self.output = output

        temp = []
        for i in range(len(self.pred)):
            total = 0
            for p_i, y_i in zip(self.pred[i], self.output[i]):
                total -= y_i * mathematics.log(p_i)
            temp.append(total)

        self.loss = sum(temp) / len(temp)
        return self.loss

    def backward(self):
        temp = []
        for i in range(len(self.pred)):
            tempRow = []
            for p_i, y_i in zip(self.pred[i], self.output[i]):
                tempRow.append((p_i - y_i) / len(self.pred))
            temp.append(Vector(tempRow))
        return Matrix(temp)

class Linear:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = None
        self.output = None

        self.weights = Matrix(input_dim, output_dim)
        self.grad = Matrix(input_dim, output_dim)

    def forward(self, x):
        self.input = x
        self.output = x @ self.weights
        return self.output

    def backward(self, dy, lr):
        dx = dy @ self.weights.transpose()

        dW = self.input.transpose() @ dy
        self.weights = self.weights - lr * dW
        self.grad = dW

        return dx

class ReLU:

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input

        temp = []
        for row in self.input:
            tempRow = []
            for x in row:
                tempRow.append(mathematics.relu(x))
            temp.append(Vector(tempRow))
        
        self.output = Matrix(temp)
        return self.output

    def backward(self, dy):
        temp = []
        for in_row, dy_row in zip(self.input, dy):
            tempRow = []
            for x_i, dy_i in zip(in_row, dy_row):
                tempRow.append(0 if x_i <= 0 else dy_i)
            temp.append(Vector(tempRow))
        return Matrix(temp)

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
        dy = self.ReLU.backward(dy)
        dy = self.l1.backward(dy, lr)
        return dy

class Softmax:

    def __init__(self):
        self.input = None
        self.output = None
        self.predicted = None
        self.grad = None

    def forward(self, input):
        self.input = input

        temp = []
        for row in input.to_list():
            sm = Vector(row).softmax()
            temp.append(sm)

        self.predicted = Matrix(temp)
        return self.predicted
    
    def backward(self, output):
        self.output = output

        temp = []
        for pred_i, out_i in zip(self.predicted, self.output):
            dot_prod = pred_i @ out_i
            tempRow = []
            for p_i, o_i in zip(pred_i, out_i):
                tempRow.append(p_i * (o_i - dot_prod))
            temp.append(Vector(tempRow))

        self.grad = Matrix(temp)
        return self.grad

class LayerNorm:

    def __init__(self, n):
        self.n = n

        self.input = None
        self.output = None
        self.mu = None
        self.var = None
        self.scale = None

        self.epsilon = 1e-6
        self.gamma = Vector([1] * n)
        self.beta = Vector([0] * n)

    def forward(self, input):
        self.input = input.to_list()

        self.mu = [mathematics.mean(row) for row in self.input]
        self.var = [mathematics.var(row) for row in self.input]
        self.scale = [(var_i + self.epsilon) ** -0.5 for var_i in self.var]

        temp = []
        for row, scale_i, mu_i in zip(self.input, self.scale, self.mu):
            tempRow = []
            for x_val, gamma, beta in zip(row, self.gamma, self.beta):
                tempRow.append(((x_val - mu_i) * scale_i) * gamma + beta)
            temp.append(Vector(tempRow))
        return Matrix(temp)

    def backward(self, dy, lr):
        n = self.n

        d_gamma = []
        d_beta = []

        for i in range(n):
            d_gamma.append(sum([(x[i] - self.mu[j]) * scale * dy[j][i] for x, scale, j in zip(self.input, self.scale, range(len(self.input)))]))
            d_beta.append(sum([dy[j][i] for j in range(len(self.input))]))

        temp = []
        for i in range(len(self.input)):
            x_hat = Vector([(x - self.mu[i]) * self.scale[i] for x in self.input[i]])
            g = Vector([dy[i][j] * self.gamma[j] for j in range(n)])

            mean_g = sum(g) / len(g)
            mean_g_xhat = sum(g[j] * x_hat[j] for j in range(n)) / n

            tempRow = Vector([self.scale[i] * (g[j] - mean_g - x_hat[j] * mean_g_xhat) for j in range(n)])
            temp.append(tempRow)

        self.grad_gamma = Vector(d_gamma)
        self.grad_beta = Vector(d_beta)

        self.gamma = self.gamma - lr * self.grad_gamma
        self.beta = self.beta - lr * self.grad_beta

        return Matrix(temp)

class SingleHeadAttention:

    def __init__(self, n_embed, head_size):
        self.input = None
        self.output = None

        self.w_queries = Linear(n_embed, head_size)
        self.w_keys = Linear(n_embed, head_size)
        self.w_values = Linear(n_embed, head_size)

        self.queries = None
        self.keys = None
        self.values = None
        self.scale = (head_size) ** -0.5

        self.sm = Softmax()

        self.d_q = None
        self.d_k = None
        self.d_v = None

    def forward(self, x):
        self.input = x

        query = self.w_queries.forward(x)
        key = self.w_keys.forward(x)
        value = self.w_values.forward(x)

        self.queries = query
        self.keys = key
        self.values = value

        self.wei = (query @ key.transpose()) * self.scale
        self.wei = self.wei.tril()

        self.output = self.sm.forward(self.wei) @ value
        return self.output

    def backward(self, dy, lr):
        dV = self.sm.predicted.transpose() @ dy
        dS = dy @ self.values.transpose()
        dWei = self.sm.backward(dS) * self.scale

        dQ = dWei @ self.keys
        dK = dWei.transpose() @ self.queries

        dx = self.w_queries.backward(dQ, lr)
        dx = dx + self.w_keys.backward(dK, lr)
        dx = dx + self.w_values.backward(dV, lr)

        return dx

class MultiHeadAttention:

    def __init__(self, n_embed, num_heads, head_size):
        self.input = None
        self.output = None

        self.heads = [SingleHeadAttention(n_embed, head_size) for _ in range(num_heads)]
        self.proj = Linear(n_embed, n_embed)

    def forward(self, input):
        self.input = input
        
        matrix_list = [h.forward(self.input) for h in self.heads]
        out = concatenate(matrix_list)
        out = self.proj.forward(out)
        
        self.output = out
        return self.output

    def backward(self, dy, lr):
        d_cat = self.proj.backward(dy, lr)
        head_size = self.heads[0].w_values.output_dim
        dx = None
        for h, head in enumerate(self.heads):
            cols = []
            for row in d_cat:
                cols.append(Vector(row.to_list()[h * head_size:(h + 1) * head_size]))
            d_head = head.backward(Matrix(cols), lr)
            dx = d_head if dx is None else dx + d_head
        return dx

class TransformerBlock:

    def __init__(self, n_embed, num_heads):
        head_size = n_embed // num_heads

        self.attention = MultiHeadAttention(n_embed, num_heads, head_size)
        self.ff = FeedForward(n_embed)

        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def forward(self, x):
        out = x + self.attention.forward(self.ln1.forward(x))
        out = out + self.ff.forward(self.ln2.forward(out))
        return out

    def backward(self, dy, lr):
        d_ff = self.ff.backward(dy, lr)
        dx = dy + self.ln2.backward(d_ff, lr)

        d_attn = self.attention.backward(dx, lr)
        dx = dx + self.ln1.backward(d_attn, lr)

        return dx

class Transformer:

    def __init__(self, vocab_size, n_embed, num_heads, block_size, n_blocks):
        self.block_size = block_size

        self.embedding = Embedding(vocab_size, n_embed)
        self.pos_encoding = PositionalEncoding(block_size, n_embed)
        self.blocks = [TransformerBlock(n_embed, num_heads) for _ in range(n_blocks)]
        self.ln = LayerNorm(n_embed)
        self.unembedding = Linear(n_embed, vocab_size)
        self.sm = Softmax()

    def forward(self, input):
        x = self.embedding.forward(input) + self.pos_encoding.forward(input)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln.forward(x)
        x = self.unembedding.forward(x)
        x = self.sm.forward(x)
        return x

    def backward(self, dy, lr):
        dx = self.unembedding.backward(dy, lr)
        dx = self.ln.backward(dx, lr)

        for block in reversed(self.blocks):
            dx = block.backward(dx, lr)

        self.embedding.backward(dx, lr)
        self.pos_encoding.backward(dx, lr)