import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric.nn as gnn
import sys 
sys.path.append('/home/80027464/TGVAE/model/')
from sublayer import * 



class EncoderLayer(nn.Module) : 
    def __init__(self, d_model, d_ff, edge_dim, e_head, dropout, gnn_mode='res') : 
        super(EncoderLayer, self).__init__()
        self.gnn_mode = gnn_mode
        self.norm1 = LayerNorm(d_model)
        self.attn1 =  gnn.GATConv(d_model, d_model//e_head, heads=e_head, dropout=dropout, edge_dim=edge_dim)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.attn2 = gnn.GATConv(d_model, d_model//e_head, heads=e_head, dropout=dropout, edge_dim=edge_dim)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout = dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, nf, ei, ew) : 
        if self.gnn_mode == 'res' : 
            nf = nf + F.leaky_relu(self.norm1(self.attn1(self.drop1(nf), ei, ew)))
            nf = nf + F.leaky_relu(self.norm2(self.attn2(self.drop2(nf), ei, ew)))
            nf = nf + F.leaky_relu(self.norm3(self.feed_forward(self.drop3(nf))))

        elif self.gnn_mode == 'res+' : 
            nf = nf + self.attn1(self.drop1(F.leaky_relu(self.norm1(nf))), ei, ew)
            nf = nf + self.attn2(self.drop2(F.leaky_relu(self.norm2(nf))), ei, ew)
            nf = nf + self.feed_forward(self.drop3(F.leaky_relu(self.norm3(nf))))

        return nf
    


class Encoder(nn.Module) :
    def __init__(self, d_model, d_ff, edge_dim, e_head, dropout, n_layer, encoder_mode='none', pool_mode='add', gnn_mode='res6_relu') : 
        super(Encoder, self).__init__()
        self.encoder_mode = encoder_mode
        self.pool_mode = pool_mode
        self.layers = clones(EncoderLayer(d_model, d_ff, edge_dim, e_head, dropout, gnn_mode), n_layer)
        self.norm = LayerNorm(d_model)

    def forward(self, nf, ei, ew, batch) : 
        if self.encoder_mode == 'dense' : 
            dense = nf 
            for layer in self.layers : 
                nf = layer(nf, ei, ew) + dense 
                dense = nf + dense
        elif self.encoder_mode == 'res' : 
            res = nf 
            for layer in self.layers : 
                nf = layer(nf, ei, ew) + res 
                res = nf 
        elif self.encoder_mode == 'none' : 
            for layer in self.layers : 
                nf = layer(nf, ei, ew)

        nf = self.norm(nf)

        if self.pool_mode == 'add' : 
            pool = gnn.global_add_pool(nf, batch)
        elif self.pool_mode == 'mean' : 
            pool = gnn.global_mean_pool(nf, batch)
        elif self.pool_mode == 'max' :
            pool = gnn.global_max_pool(nf, batch)            

        return pool
    



class LatentModel(nn.Module) : 
    def __init__(self, d_model) : 
        super(LatentModel, self).__init__()
        self.mu = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, d_model)

    def forward(self, x) : 
        mu, sigma = self.mu(x), self.sigma(x)
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(0.5 * sigma) * eps
        return z, mu, sigma




class DecoderLayer(nn.Module) : 
    def __init__(self, d_model, d_ff, d_head, dropout) : 
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_head, d_model, dropout) 
        self.src_attn = MultiHeadedAttention(d_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
    
    def forward(self, x, memory, smi_mask) : 
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, smi_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, memory, memory, None))
        return self.sublayer[2](x, self.feed_forward)





class Decoder(nn.Module) :
    def __init__(self, d_model, d_ff, d_head, dropout, n_layer) : 
        super(Decoder, self).__init__() 
        self.layers = clones(DecoderLayer(d_model, d_ff, d_head, dropout), n_layer)
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, memory, smi_mask) : 
        for layer in self.layers : 
            x = layer(x, memory, smi_mask)
        return self.norm(x)
    



class TGVAE(nn.Module) : 
    def __init__(self,
                 d_model=[512, 512],
                 d_ff=1024, 
                 edge_dim=None,
                 n_head=[1, 16],
                 dropout=0.5,
                 n_layer=[8, 8],
                 node_vocab_size=0,
                 smi_vocab_size=0,
                 edge_vocab_size=4,
                 encoder_mode='dense',
                 pool_mode='add',
                 gnn_mode='res6_relu') : 
        
        super(TGVAE, self).__init__()
        self.d_model = d_model

        self.node_vocab_size = node_vocab_size
        self.edge_vocab_size = edge_vocab_size

        self.node_embedding = gnn.GATConv(node_vocab_size, d_model[0]//n_head[0], n_head[0], dropout = dropout)
        self.norm = LayerNorm(d_model[0])
        self.smi_embedding = nn.Sequential(Embeddings(d_model[1], smi_vocab_size), PositionalEncoding(d_model[1], dropout))


        self.encoder = Encoder(d_model[0], d_ff, edge_dim, n_head[0], dropout, n_layer[0], encoder_mode, pool_mode, gnn_mode)
        self.latent_model = LatentModel(d_model[0]) 
        self.decoder = Decoder(d_model[1], d_ff, n_head[1], dropout, n_layer[1])


        self.generator = nn.Linear(d_model[1], smi_vocab_size)



    def inference(self, z, smi, smi_mask) : 
        smi = self.smi_embedding(smi)
        out = self.decoder(smi, z, smi_mask)
        out = F.log_softmax(self.generator(out), dim = -1)
        return out
    
    
    def forward(self, graph, smi, smi_mask) : 
        node_feature, edge_index, edge_weight, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch

        node_feature = (F.one_hot(node_feature, num_classes=self.node_vocab_size)).float()
        edge_weight = (F.one_hot(edge_weight, num_classes=self.edge_vocab_size)).float()

        node_feature = F.leaky_relu(self.node_embedding(node_feature, edge_index, edge_weight))

        smi = self.smi_embedding(smi)

        pool = self.encoder(node_feature, edge_index, edge_weight, batch)

        z, mu, sigma = self.latent_model(pool)
        
        out = self.decoder(smi, z, smi_mask)

        out = F.log_softmax(self.generator(out), dim = -1)

        return out, mu, sigma
    
