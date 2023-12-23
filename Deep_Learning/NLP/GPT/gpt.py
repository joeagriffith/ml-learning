import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
batch_size = 64
block_size = 256
num_epochs = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------

torch.manual_seed(1337)

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

def get_batch(val=False):
    data = train_data if not val else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            val = split == 'val'
            X, Y = get_batch(val)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

class SelfAttention(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.Wk = nn.Linear(n_embd, head_size, bias=False)
        self.Wq = nn.Linear(n_embd, head_size, bias=False)
        self.Wv = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B,T,C)
        B, T, C = x.shape

        k = self.Wk(x) # (B,T,head_size)
        q = self.Wq(x) # (B,T,head_size)
        v = self.Wv(x)

        scores = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        out = scores @ v
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x

class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[AttentionBlock(n_embd, n_head) for _ in range(n_layer)]) 
        self.layernorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embeddings + pos_embedding
        x = self.blocks(x)
        x = self.layernorm(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = GPTLanguageModel().to(device)
train_losses = []
val_losses = []
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loop = tqdm(enumerate(range(num_epochs)), leave=False)
for i, steps in loop:

    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

    if i % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f'step {i}: train loss {losses["train"]}, val loss {losses["val"]}')

    loop.set_description(f'Epoch: [{i}/{num_epochs}]')
    if i > 0:
        loop.set_postfix(
            train_loss = train_losses[-1],
            val_loss = val_losses[-1]
        )


idx = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend()
plt.show()