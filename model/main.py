import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric.nn as gnn
from model.sublayer import * 



class EncoderLayer(nn.Module) : 
    def __init__(self, dim, dim_ff, dim_edge, num_head, dropout) : 
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn1 =  gnn.GATConv(dim, dim//num_head, heads=num_head, dropout=dropout, edge_dim=dim_edge)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.attn2 = gnn.GATConv(dim, dim//num_head, heads=num_head, dropout=dropout, edge_dim=dim_edge)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = LayerNorm(dim)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, nf, ei, ew) : 
        nf = nf + F.leaky_relu(self.norm1(self.attn1(self.drop1(nf), ei, ew)))
        nf = nf + F.leaky_relu(self.norm2(self.attn2(self.drop2(nf), ei, ew)))
        nf = nf + F.leaky_relu(self.norm3(self.feed_forward(self.drop3(nf))))
        return nf
    


class Encoder(nn.Module) :
    def __init__(self, dim, dim_ff, dim_edge, num_head, num_layer, dropout) : 
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(dim, dim_ff, dim_edge, num_head, dropout), num_layer)
        self.norm = LayerNorm(dim)

    def forward(self, nf, ei, ew, batch) : 
        dense = nf 
        for layer in self.layers : 
            nf = layer(nf, ei, ew) + dense 
            dense = nf + dense
        nf = self.norm(nf)
        return gnn.global_add_pool(nf, batch)
    



class LatentModel(nn.Module) : 
    def __init__(self, dim) : 
        super(LatentModel, self).__init__()
        self.mu = nn.Linear(dim, dim)
        self.sigma = nn.Linear(dim, dim)

    def forward(self, x) : 
        mu, sigma = self.mu(x), self.sigma(x)
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(0.5 * sigma) * eps
        return z, mu, sigma



class DecoderLayer(nn.Module) : 
    def __init__(self, dim, dim_ff, num_head, dropout) : 
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_head, dim, dropout) 
        self.src_attn = MultiHeadedAttention(num_head, dim, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = clones(SublayerConnection(dim, dropout), 3)
    
    def forward(self, x, memory, smi_mask) : 
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, smi_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, memory, memory, None))
        return self.sublayer[2](x, self.feed_forward)





class Decoder(nn.Module) :
    def __init__(self, dim, dim_ff, num_head, num_layer, dropout) : 
        super(Decoder, self).__init__() 
        self.layers = clones(DecoderLayer(dim, dim_ff, num_head, dropout), num_layer)
        self.norm = LayerNorm(dim)
    
    def forward(self, x, memory, smi_mask) : 
        for layer in self.layers : 
            x = layer(x, memory, smi_mask)
        return self.norm(x)
    



class TGVAE(nn.Module) : 
    def __init__(self,
                 dim_encoder=512,
                 dim_decoder=512,
                 dim_encoder_ff=1024,
                 dim_decoder_ff=1024,
                 dim_edge=512,
                 num_encoder_layer=8,
                 num_decoder_layer=8,
                 num_encoder_head=1,
                 num_decoder_head=16,
                 dropout_encoder=0.5,
                 dropout_decoder=0.5,
                 size_graph_vocab=100,
                 size_smi_vocab=100,
                 size_edge_vocab=4,
                 device='cpu') : 
        
        super(TGVAE, self).__init__()

        self.device = device
        self.size_graph_vocab = size_graph_vocab
        self.size_edge_vocab = size_edge_vocab

        self.edge_embedding = Embeddings(dim_encoder, size_edge_vocab)
        self.graph_embedding = GraphEmbedding(size_graph_vocab, dim_encoder, num_encoder_head, dropout_encoder)
        self.smi_embedding = SmilesEmbedding(size_smi_vocab, dim_decoder, dropout_decoder)

        self.encoder = Encoder(dim_encoder, dim_encoder_ff, dim_edge, num_encoder_head, num_encoder_layer, dropout_encoder)
        self.latent_model = LatentModel(dim_encoder) 
        self.decoder = Decoder(dim_decoder, dim_decoder_ff, num_decoder_head, num_decoder_layer, dropout_decoder)

        self.generator = nn.Linear(dim_decoder, size_smi_vocab)

    def generate_step(self, z, smi, smi_mask) : 
        smi = self.smi_embedding(smi)
        out = self.decoder(smi, z, smi_mask)
        out = F.log_softmax(self.generator(out), dim=-1)
        _, idx = torch.topk(out, 1, dim=-1)
        return idx[:, -1, :] 
    
    
    def generate(self, config, num_gen) : 
        z = torch.randn(num_gen, config.dim_encoder).to(self.device)
        smi = torch.zeros(num_gen, 1, dtype=torch.long).to(self.device) # [START] Token
        for _ in range(config.max_token - 1) : 
            smi_next_token = self.generate_step(z, smi, get_mask(smi, config.vocab_smi).to(self.device))
            smi = torch.cat([smi, smi_next_token], dim=-1)
        torch.cuda.empty_cache()
        return smi 

    def forward(self, graph, smi, smi_mask) : 
        node_feature, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch

        node_feature = self.graph_embedding(node_feature, edge_index, edge_attr)
        edge_attr = self.edge_embedding(edge_attr)
        smi = self.smi_embedding(smi)

        pool = self.encoder(node_feature, edge_index, edge_attr, batch)
        z, mu, sigma = self.latent_model(pool)
        out = self.decoder(smi, z, smi_mask)
        out = F.log_softmax(self.generator(out), dim = -1)

        return out, mu, sigma
    