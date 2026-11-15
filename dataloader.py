import mathematics
from tensor import Vector, Matrix
from Transformer import Transformer, CrossEntropyLoss

sample_txt = "The sun is a mass"

vocab_dict = {}
idx_dict = {}

for c in sample_txt:
    if c not in vocab_dict:
        vocab_dict[c] = len(vocab_dict)
        idx_dict[len(vocab_dict)-1] = c

with open('poc-corpus.txt', 'r', encoding='utf-8') as f:
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

sample_txt = text
parsed_txt = word_parser(sample_txt)

words = sorted(list(set(parsed_txt)))
vocab_size = len(words)

def one_hot_generator(vec, size):
    temp = []
    zero_vec = [0] * size
    for elem in vec:
        vec_copy = zero_vec.copy()
        vec_copy[elem] = 1
        temp.append(vec_copy)
    return temp

def sm_to_idx(vec):
    vec = vec.to_list()
    max_elem = max(vec)
    return vec.index(max_elem)

def smat_to_idx(mat):
    temp = []
    for vec in mat:
        temp.append(sm_to_idx(vec))
    return temp

class LanguageModel:

    def __init__(self, corpus):
        self.corpus = corpus
        self.words = sorted(set(corpus.split()))
        
        self.stoi = { word:i for i, word in enumerate(self.words) }
        self.itos = { i:word for i, word in enumerate(self.words) }
        self.encode = lambda s: [self.stoi[word] for word in s.split()] # String to list of numbers
        self.decode = lambda ids: ' '.join(self.itos[i] for i in ids) # List of numbers to string
        
        self.vocab_size = len(self.words)
        self.block_size = 16
        n_embed = 16
        num_heads = 4
        n_blocks = 1

        self.T = Transformer(self.vocab_size, n_embed, num_heads, self.block_size, n_blocks)
        self.CEL = CrossEntropyLoss()
        self.output = None

    def generate(self, prompt, num_new_tokens):
        ids = self.encode(prompt)

        for token in range(num_new_tokens):
            context = ids[-self.block_size:]
            probs = self.T.forward(context)

            next_id = probs[-1].argmax()
            ids.append(next_id)
        self.output = self.decode(ids)
        return self.output

    def train(self, num_epochs, lr):
        ids = self.encode(self.corpus)
        block_size = self.block_size

        assert len(ids) > block_size

        for epoch in range(1, num_epochs+1):
            r = mathematics.random_int(0, len(ids) - block_size - 1)
            
            x = ids[r : r + block_size]
            targets = ids[r + 1 : r + block_size + 1]
            
            y = one_hot_generator(targets, self.vocab_size)
            
            output = self.T.forward(x)
            loss = self.CEL.forward(output, y)
            
            self.T.backward(self.CEL.backward(), lr)
            
            if(epoch % 200 == 0):
                print(f'epoch {epoch}: loss = {loss}')

m = LanguageModel(sample_txt)
print(m.generate("gravity", 10))
m.train(2000, 0.25)

print(m.generate("force equals", 3))
print(m.generate("gravity pulls", 3))
print(m.generate("the ball", 3))