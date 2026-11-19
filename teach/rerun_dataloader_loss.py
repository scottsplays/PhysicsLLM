# Re-run of dataloader.py's LanguageModel.train setup with loss logged every step.
# Same corpus, vocab, model shape, lr and loop as dataloader.py; dataloader.py is not imported.
import json, sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0])
import mathematics
from Transformer import Transformer, CrossEntropyLoss
np.random.seed(0)
text = open('poc-corpus.txt', encoding='utf-8').read()
words = sorted(set(text.split())); stoi = {w: i for i, w in enumerate(words)}
ids = [stoi[w] for w in text.split()]
V, B = len(words), 16
T = Transformer(V, 16, 4, B, 1); CEL = CrossEntropyLoss()
def onehot(t): return [[1 if j == k else 0 for j in range(V)] for k in t]
# epoch 0: loss of the untrained model, averaged over 20 random windows
rng = np.random.RandomState(1); l0 = []
for _ in range(20):
    r = int(rng.randint(0, len(ids) - B - 1)); l0.append(CEL.forward(T.forward(ids[r:r+B]), onehot(ids[r+1:r+B+1])))
losses = []
for epoch in range(1, 2001):
    r = mathematics.random_int(0, len(ids) - B - 1)
    out = T.forward(ids[r:r+B]); loss = CEL.forward(out, onehot(ids[r+1:r+B+1]))
    T.backward(CEL.backward(), 0.25); losses.append(float(loss))
print(json.dumps({"V": V, "epoch0": float(np.mean(l0)), "losses": losses}))
