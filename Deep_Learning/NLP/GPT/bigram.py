import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 1000

with open('../Datasets/mini_shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from character to integer and a reverse mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long).to(device)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

torch.manual_seed(42)
batch_size = 4
block_size = 8

def get_batch(val=False):
    data = train_data if not val else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel(vocab_size).to(device)
train_losses = []
val_losses = []
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)

def eval(model, n=30):
    losses = []
    with torch.no_grad():
        for _ in range(10):
            xb, yb = get_batch(val=True)
            _, loss = model(xb, yb)
            losses.append(loss.item())

    return sum(losses) / len(losses)



loop = tqdm(enumerate(range(num_epochs)), leave=False)
for i, steps in loop:

    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
    train_losses.append(loss.item())

    val_losses.append(eval(model))

    loop.set_description(f'Epoch: [{i}/{num_epochs}]')
    loop.set_postfix(
        train_loss = loss.item(),
        val_loss = val_losses[-1]
    )


idx = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend()
plt.show()