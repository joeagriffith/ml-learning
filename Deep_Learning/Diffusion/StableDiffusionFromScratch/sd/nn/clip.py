import torch
import torch.nn as nn
import torch.nn.functional as F
from Deep_Learning.Diffusion.StableDiffusionFromScratch.sd.nn.parts import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (Batch_Size, Seq_Len)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        token_embedding = self.token_embedding(tokens)

        return token_embedding + self.position_embedding


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)

        residual = x

        ## SELF ATTENTION

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residual

        ## MLP

        residual = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
        x = self.linear_2(x)
        x += residual

        return x


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output

