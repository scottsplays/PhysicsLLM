import mathematics
from tensor import Vector, Matrix
from layer import Layer, Weight

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def word_parser(txt):
    words = []
    word = ""
    for i, c in enumerate(txt):
        if(c == " "):
            words.append(word)
            word = ""
        else:
            word += c
        if(i == len(txt) - 1):
            words.append(word)
    return words

class Embedding:

    def __init__(self, n_tokens, embed_size):
        self.embedding_table = Weight(n_tokens, embed_size, bias = False)

    def forward(self, x):
        return Layer(self.embedding_table[x])

E = Embedding(65, 32)

sample_txt = text
parsed_txt = word_parser(sample_txt)

words = sorted(list(set(parsed_txt)))
vocab_size = len(words)

stoi = { ch:i for i, ch in enumerate(words) }
itos = { i:ch for i, ch in enumerate(words) }
encode = lambda s: [stoi[word] for word in word_parser(s)] # String to list of numbers
decode = lambda l: ' '.join([itos[i] for i in l]) # List of numbers to string

data = Vector(encode(text))

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = Vector(mathematics.random_ints(batch_size, 0, len(data) - block_size))
  x = Matrix([data[i:i+block_size] for i in ix])
  y = Matrix([data[i+1:i+block_size+1] for i in ix])
  return x, y



n_embd = 192 # Was 384
n_layer = 3 # Was 6
dropout = 0.2
batch_size = 32 # Was 64
block_size = 128 # Was 256
learning_rate = 3e-4

class Head:

  def __init__(self, head_size):
    self.key = Weight(n_embd, head_size, bias=False)
    self.query = Weight(n_embd, head_size, bias=False)
    self.value = Weight(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention:

  def __init__(self, num_heads, head_size):
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = Weight(n_embd, n_embd)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward:

  def __init__(self, n_embd):
    self.net = nn.Sequential(
      Weight(n_embd, 4 * n_embd),
      nn.ReLU(),
      Weight(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block:

  def __init__(self, n_embd, n_head):
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel:

  def __init__(self):
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = Weight(n_embd, vocab_size)

  def forward(self, idx, targets = None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

m = BigramLanguageModel()