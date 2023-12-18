import torch
import torch.nn as nn
import torch.nn.functional as F
from Deep_Learning.Diffusion.StableDiffusionFromScratch.nn.parts import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (Batch_Size, Seq_Len)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        token_embedding = self.token_embedding(tokens)

        # # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # position_embedding = self.position_embedding.unsqueeze(0).repeat(tokens.shape[0], 1, 1)

        # # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # return token_embedding + position_embedding

        return token_embedding + self.position_embedding


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)

        residual = x

        ## SELF ATTENTION

        x = self.layer_norm_1(x)
        x = self.attention(x)
        x += residual

        ## MLP

        residual = x

        x = self.layer_norm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
        x = self.linear_2(x)
        x += residual

        return x


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layer_norm(state)

        return output

