import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
import math 
import copy 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

###### Utils Layer ######
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

###### Encoder Layer ######
class EncoderLayer(nn.Module) : 
    def __init__(self, d_model, n_heads, feed_forward, dropout) : 
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.norm1 = LayerNorm(d_model) 
        self.attn = gnn.GATConv(d_model, d_model // n_heads, n_heads, dropout = dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.feed_forward = feed_forward
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, nf, ei) : 
        nf = nf + self.drop1(self.attn(self.norm1(nf), ei))
        nf = nf + self.drop2(self.feed_forward(self.norm2(nf)))
        return nf 
    
class Encoder(nn.Module) : 
    def __init__(self, layer, N, d_latent) : 
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.mu = nn.Linear(layer.d_model, d_latent)
        self.sigma = nn.Linear(layer.d_model, d_latent)

    def get_z(self, mu, sigma) : 
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(0.5 * sigma) * eps
        return z 
    
    def forward(self, nf, ei, batch) : 
        for layer in self.layers : 
            nf = layer(nf, ei)
        nf = self.norm(nf) 
        pool = gnn.global_add_pool(nf, batch)
        mu, sigma = self.mu(pool), self.sigma(pool)
        z = self.get_z(mu, sigma)
        return z, mu, sigma 


###### Decoder Layer ######
class DecoderLayer(nn.Module) : 
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout) : 
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn 
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask) : 
        m = memory 
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
class Decoder(nn.Module):
    def __init__(self, layer, N, d_latent):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.upsize = nn.Sequential(
            nn.Linear(d_latent, layer.d_model),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(layer.d_model, layer.d_model)
        )
    def forward(self, x, memory, src_mask, tgt_mask):
        memory = F.relu(self.upsize(memory))
        for layer in self.layers:
            x = layer(x, memory, None, tgt_mask)
        return self.norm(x)
    

######## Model ########
class Transformer(nn.Module):
    def __init__(self, d_model, d_latent, d_ff, num_head, num_layer, dropout, vocab, gvocab) : 
        super(Transformer, self).__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(num_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, num_head, c(ff), dropout), num_layer, d_latent)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layer, d_latent)

        self.src_embedding = nn.Embedding(len(gvocab), d_model)
        self.tgt_embedding = nn.Sequential(Embeddings(d_model, len(vocab)), c(position))
        
        self.generator = nn.Linear(d_model, len(vocab))

    def inference(self, src_z, tgt, src_mask, tgt_mask) :
        tgt = self.tgt_embedding(tgt)
        out = self.decoder(tgt, src_z, src_mask, tgt_mask)
        out = F.log_softmax(self.generator(out), dim = -1)
        return out
        
    def forward(self, src, tgt, src_mask, tgt_mask):  
        nf, ei, batch = src.x, src.edge_index, src.batch
        nf = self.src_embedding(nf)
        tgt = self.tgt_embedding(tgt)
        z, mu, sigma = self.encoder(nf, ei, batch)
        out = self.decoder(tgt, z, src_mask, tgt_mask)
        out = F.log_softmax(self.generator(out), dim = -1)

        return out, mu, sigma