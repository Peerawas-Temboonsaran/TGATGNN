import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool
from torch_geometric.nn       import GATConv
from torch_scatter            import scatter_add
from torch_geometric.nn.inits import glorot, zeros

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

torch.cuda.empty_cache()

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.2):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Apply dropout after activation
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)  # Apply dropout after the second layer
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class GI_M1_Transformer(nn.Module):
    def __init__(self, neurons, num_heads=8, transformer_layers=4, dropout_rate=0.2):
        super(GI_M1_Transformer, self).__init__()
        self.input_layer = nn.Linear(neurons + 103, neurons)  # Include elemental composition
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(neurons, num_heads, ff_hidden_dim=neurons * 2, dropout=dropout_rate) for _ in range(transformer_layers)]
        )
        self.attention_layer = nn.Linear(neurons, 1)
    
    def forward(self, x, batch, global_feat):
        counts = torch.unique(batch, return_counts=True)[-1]
        graph_embed = torch.repeat_interleave(global_feat, counts, dim=0)
        chunk = torch.cat([x, graph_embed], dim=-1)
        x = F.relu(self.input_layer(chunk))

        # Replaces the MLP
        '''x = x.unsqueeze(1)  
        
        for layer in self.transformer_layers:
            x = layer(x)
    
        x = x.squeeze(1)
        attention_scores = self.attention_layer(x)
        weights = softmax(attention_scores, batch)'''

        # Combine Transformer outputs with original MLP-based GI-M1
        x_transformed = x.unsqueeze(1)  
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed)
        x_transformed = x_transformed.squeeze(1)
        attention_scores = self.attention_layer(x)
        attention_scores_transformed = self.attention_layer(x_transformed)
        weights = softmax((attention_scores + attention_scores_transformed) / 2, batch)
        
        return weights

class CLUSTER_Attention(nn.Module):
    def __init__(self,neurons_1,neurons_2,num_cluster, cluster_method = 'random'):
        '''
        Global-Attention Mechanism based on clusters (position grouping) of crystals' elements
        > Defined in paper as *GI-M2*, *GI-M3*, *GI-M4*
        ======================================================================================
        neurons_1       : number of neurons to use for layer_1
        neurons_2       : number of neurons to use for the attention-layer
        num_cluster     : number of clusters to use 
        cluster_method  : unpooling method to use 
            - fixed     : (GI-M2)
            - random    : (GI-M3)
            - learnable : (GI-M4)
        '''        
        super(CLUSTER_Attention,self).__init__()
        self.learn_unpool       = Linear(2*neurons_1+3,num_cluster)        
        self.layer_1            = Linear(neurons_1,neurons_2)
        self.negative_slope     = 0.45
        self.atten_layer        = Linear(neurons_2,1)
        self.clustering_method  = cluster_method
        if not self.training: np.random.seed(0)

    
    def forward(self,x,cls,batch):
        r_x     = self.unpooling_featurizer(x,cls,batch)
        r_x     = F.leaky_relu(self.layer_1(r_x.float()),self.negative_slope)
        r_x     = self.atten_layer(r_x)
        weights  = softmax(r_x,batch)
        return weights

    def unpooling_featurizer(self,x,cls,batch):
        g_counts  = torch.unique(batch,return_counts=True)[-1].tolist()
        split_x   = torch.split(x,g_counts)
        split_cls = torch.split(cls,g_counts)
        new_x     = torch.tensor([]).cuda()
        # break batch into graphs
        for i in range(len(split_x)):
            graph_features = split_x[i]
            clus_t         = split_cls[i].view(-1)
            cluster_sum    = scatter_add(graph_features,clus_t,0)
            zero_sum       = torch.zeros_like(cluster_sum)
            if   len(graph_features) == 1: 
                new_x = torch.cat([new_x,cluster_sum],dim=0)
            elif len(graph_features) == 2:
                new_x = torch.cat([new_x,cluster_sum,cluster_sum],dim=0)
            else:
                region_arr  = np.array(clus_t.tolist())
                # choosing unpooling method
                if self.clustering_method == 'fixed':        #--- GI M-2
                    random_sets  = clus_t.tolist()                
                elif self.clustering_method   == 'random':   #--- GI M-3
                    random_sets  = [np.random.choice(np.setdiff1d(region_arr,e)) for e in region_arr]
                elif self.clustering_method == 'learnable':  #--- GI M-4
                    total_feat   = graph_features.sum(dim=0).unsqueeze(0)
                    region_input = torch.cat([graph_features,total_feat,clus_t.unsqueeze(0).float()],dim=-1)
                    random_sets  = torch.argmax(F.softmax(self.learn_unpool(region_input)),dim=1).tolist()
                
                # normalized-regions
                unique, counts = np.unique(region_arr, return_counts=True) 
                counts         = counts/counts.sum()
                sets_dict      = dict(zip(unique, counts))             
                random_ratio   = torch.tensor([sets_dict[i] for i in random_sets]).cuda()
                random_ratio   = (random_ratio / random_ratio.sum()).view(-1, 1)
                
                cluster_sum = cluster_sum[random_sets]
                cluster_sum = cluster_sum*random_ratio
                new_x       = torch.cat([new_x,cluster_sum],dim=0)
        return new_x        

class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False,
                 dropout=0.1, bias=True, **kwargs):
        '''
        Our Augmented Graph Attention Layer
        > Defined in paper as *AGAT*
        =======================================================================
        in_features    : input-features
        out_features   : output-features
        edge_dim       : edge-features
        heads          : attention-heads
        concat         : to concatenate the attention-heads or sum them
        dropout        : 0
        bias           : True
        '''    
        super(GAT_Crystal, self).__init__(aggr='add',flow='target_to_source', **kwargs)
        self.in_features       = in_features
        self.out_features      = out_features
        self.heads             = heads
        self.concat            = concat
        self.dropout           = dropout
        self.neg_slope         = 0.2
        self.prelu             = nn.PReLU()
        self.bn1               = nn.BatchNorm1d(heads)
        self.W                 = Parameter(torch.Tensor(in_features+edge_dim,heads*out_features))
        self.att               = Parameter(torch.Tensor(1,heads,2*out_features))

        if bias and concat       : self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat : self.bias = Parameter(torch.Tensor(out_features))
        else                     : self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x,edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i,edge_attr): 
        x_i   = torch.cat([x_i,edge_attr],dim=-1)
        x_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        x_i   = F.softplus(torch.matmul(x_i,self.W))
        x_j   = F.softplus(torch.matmul(x_j,self.W))
        x_i   = x_i.view(-1, self.heads, self.out_features)
        x_j   = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))
        alpha = softmax(alpha,edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j   = (x_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
        return x_j

    def update(self, aggr_out,x):
        if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:                      aggr_out = aggr_out.mean(dim=0)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out

class GATGNN(nn.Module):
    def __init__(self, heads, classification=None, neurons=64, nl=3, xtra_layers=True, global_attention='composition',
                 unpooling_technique='random', concat_comp=False, edge_format='CGCNN'):
        super(GATGNN, self).__init__()

        self.n_heads = heads
        self.classification = classification is not None
        self.unpooling = unpooling_technique
        self.g_a = global_attention
        self.number_layers = nl
        self.concat_comp = concat_comp
        self.additional = xtra_layers

        self.neurons = neurons
        self.embed_n = Linear(92, neurons)
        self.embed_e = Linear(41, neurons) if edge_format in ['CGCNN', 'NEW'] else Linear(9, neurons)
        self.embed_comp = Linear(103, neurons)

        self.node_att = nn.ModuleList([GAT_Crystal(neurons, neurons, neurons, self.n_heads) for _ in range(nl)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(neurons) for _ in range(nl)])

        if self.g_a == 'composition':
            self.global_attention_layer = GI_M1_Transformer(neurons)
        
        if self.concat_comp:
            reg_h = neurons * 2
        else:
            reg_h = neurons

        if self.additional:
            self.linear1 = nn.Linear(reg_h, reg_h)
            self.linear2 = nn.Linear(reg_h, reg_h)

        self.out = Linear(reg_h, 2) if self.classification else Linear(reg_h, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch, global_feat = data.batch, data.global_feature

        x = self.embed_n(x)
        edge_attr = F.leaky_relu(self.embed_e(edge_attr), 0.2)
        
        layer_embeddings = []
        for layer, bn in zip(self.node_att, self.batch_norm):
            x = layer(x, edge_index, edge_attr)
            x = bn(x)
            x = F.softplus(x)
            layer_embeddings.append(x.clone().detach())  # Store embeddings
        
        if self.g_a == 'composition':
            weights = self.global_attention_layer(x, batch, global_feat)
            x = x * weights
        
        y = global_add_pool(x, batch)
        if self.concat_comp:
            y = torch.cat([y, F.leaky_relu(self.embed_comp(global_feat), 0.2)], dim=-1)
        
        if self.additional:
            y = F.softplus(self.linear1(y))
            y = F.softplus(self.linear2(y))
        
        y = self.out(y).squeeze()
        return y, layer_embeddings