from tgatgnn.data                   import *
import tgatgnn.TGATGNN_1            as TG1
import tgatgnn.TGATGNN_2            as TG2
import tgatgnn.TGATGNN_3            as TG3
import tgatgnn.TGATGNN_4            as TG4
from tgatgnn.pytorch_early_stopping import *
from tgatgnn.file_setter            import use_property
from tgatgnn.utils                  import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F

# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN')
parser.add_argument('--property', default='bulk-modulus',
                    choices=['absolute-energy','band-gap','bulk-modulus',
                             'fermi-energy','formation-energy',
                             'poisson-ratio','shear-modulus','new-property'],
                    help='material property to train (default: bulk-modulus)')
parser.add_argument('--data_src', default='CGCNN',choices=['CGCNN','MEGNET','NEW'],
                    help='selection of the materials dataset to use (default: CGCNN)')

# MOST CRUCIAL MODEL PARAMETERS 
parser.add_argument('--model_name', default='TGATGNN-1',
                    choices=['TGATGNN-1', 'TGATGNN-2', 'TGATGNN-3', 'TGATGNN-4'],
                    help='Choose which TGATGNN model variant to use (default: TGATGNN-1)')
parser.add_argument('--num_layers',default=5, type=int, 
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons',default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')
parser.add_argument('--num_heads',default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--use_hidden_layers',default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--global_attention',default='composition', choices=['composition','cluster']
                    ,help='selection of the unpooling method as referenced in paper GI M-1 to GI M-4 (default:composition)')
parser.add_argument('--cluster_option',default='fixed', choices=['fixed','random','learnable'],
                    help='selection of the cluster unpooling strategy referenced in paper GI M-1 to GI M-4 (default: fixed)')
parser.add_argument('--concat_comp',default=False, type=bool,
                    help='option to re-use vector of elemental composition after global summation of crystal feature.(default: False)')
parser.add_argument('--train_size',default=0.8, type=float,
                    help='ratio size of the training-set (default:0.8)')
args = parser.parse_args(sys.argv[1:])

# GATGNN --- parameters
crystal_property                      = args.property
data_src                              = args.data_src
source_comparison, training_num,RSM   = use_property(crystal_property,data_src)
norm_action, classification           = set_model_properties(crystal_property)
if training_num == None: training_num = args.train_size

number_layers                        = args.num_layers
number_neurons                       = args.num_neurons
n_heads                              = args.num_heads
xtra_l                               = args.use_hidden_layers 
global_att                           = args.global_attention
attention_technique                  = args.cluster_option
concat_comp                          = args.concat_comp
model_name                           = args.model_name


# SETTING UP CODE TO RUN ON GPU
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# DATA PARAMETERS
random_num          =  456;random.seed(random_num)

# MODEL HYPER-PARAMETERS
num_epochs      = 300
learning_rate   = 5e-3
batch_size      = 256

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':256, 'shuffle': True}

# DATALOADER/ TARGET NORMALIZATION
src_CIF         = 'CIF-DATA_NEW' if data_src == 'NEW' else 'CIF-DATA'
dataset         = pd.read_csv(f'DATA/{src_CIF}/id_prop.csv',names=['material_ids','label']).sample(frac=1,random_state=random_num)
NORMALIZER      = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA    = CIF_Dataset(dataset, root_dir = f'DATA/{src_CIF}/',**RSM)
idx_list        = list(range(len(dataset)))
random.shuffle(idx_list)

train_idx,test_val = train_test_split(idx_list,train_size=training_num,random_state=random_num)
_,       val_idx   = train_test_split(test_val,test_size=0.5,random_state=random_num)

training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm_action,df=dataset,src=data_src)
validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm_action,  df=dataset,src=data_src)

if model_name == 'TGATGNN-1':
    net_class = TG1.TransformerGATGNN
    net = net_class(n_heads, classification, neurons=number_neurons, nl=number_layers, 
                    xtra_layers=xtra_l, global_attention=global_att, 
                    unpooling_technique=attention_technique, concat_comp=concat_comp, 
                    edge_format=data_src).to(device)

elif model_name == 'TGATGNN-2':
    net_class = TG2.GATGNN
    net = net_class(n_heads, classification, neurons=number_neurons, nl=number_layers, 
                      xtra_layers=xtra_l, global_attention='composition',
                      unpooling_technique=attention_technique, concat_comp=concat_comp, edge_format=data_src).to(device)

elif model_name == 'TGATGNN-3':
    net_class = TG3.GATGNN
    net = net_class(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                                      unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src).to(device)

elif model_name == 'TGATGNN-4':
    net_class = TG4.GATGNN
    transformer_heads = 4
    num_transformer_layers = 3
    net = net_class(n_heads, transformer_heads, classification, neurons=number_neurons, nl=number_layers,
             xtra_layers=xtra_l, global_attention=global_att, unpooling_technique=attention_technique,
             concat_comp=concat_comp, edge_format=data_src, num_transformer_layers=num_transformer_layers).to(device)

else:
    raise ValueError(f"Invalid model name: {model_name}")

# LOSS & OPTMIZER & SCHEDULER
if classification == 1: criterion   = nn.CrossEntropyLoss().cuda(); funct = torch_accuracy
else                  : criterion   = nn.SmoothL1Loss().cuda()    ; funct = torch_MAE
optimizer         = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-1)
scheduler         = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)

# EARLY-STOPPING INITIALIZATION
early_stopping = EarlyStopping(patience=stop_patience, increment=1e-6,verbose=True,save_best=True,classification=classification)

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct,device)

# Cosine similarity calculation
def compute_cosine_similarity(embeddings, edge_index):
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    normalized = embeddings / (norm + 1e-6)
    N = normalized.size(0)

    # Neighbor similarity (มี edge)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    neighbor_sim = F.cosine_similarity(normalized[source_nodes], normalized[target_nodes]).mean().item()

    # All-pair similarity
    sim_matrix = torch.matmul(normalized, normalized.T)
    upper_triangle = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    all_pair_sim = sim_matrix[upper_triangle].mean().item()

    # Non-neighbor similarity
    adj = torch.zeros((N, N), device=embeddings.device)
    adj[source_nodes, target_nodes] = 1
    adj[target_nodes, source_nodes] = 1  # undirected

    non_neighbor_mask = (adj == 0) & upper_triangle
    non_neighbor_sim = sim_matrix[non_neighbor_mask].mean().item()

    return neighbor_sim, non_neighbor_sim, all_pair_sim

print(f'> TRAINING MODEL ...')
train_loader   = torch_DataLoader(dataset=training_set,   **train_param)
valid_loader   = torch_DataLoader(dataset=validation_set, **valid_param) 

all_epoch_similarities = {} 

for epoch in range(num_epochs):
    # TRAINING-STAGE
    net.train()       
    start_time       = time.time()

    for data in train_loader:
        data         = data.to(device)
        predictions, layer_embeddings  = net(data)
        train_label  = metrics.set_label('training',data)
        loss         = metrics('training',predictions,train_label,1)
        _            = metrics('training',predictions,train_label,2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.training_counter+=1
    metrics.reset_parameters('training',epoch)
    # VALIDATION-PHASE
    net.eval()
    num_val_batches = 0

    # Ensure that epoch_layer_similarities is initialized for each epoch and layer
    epoch_layer_similarities = {layer_idx: [0.0, 0.0, 0.0] for layer_idx in range(number_layers)}

    for data in valid_loader:
        data = data.to(device)
        with torch.no_grad():
            predictions, val_layer_embeddings    = net(data)
        valid_label        = metrics.set_label('validation',data)
        _                  = metrics('validation',predictions,valid_label,1)
        _                  = metrics('validation',predictions, valid_label,2)

        metrics.valid_counter+=1

        for layer_idx, embeddings in enumerate(val_layer_embeddings):
            # Compute cosine similarity for each layer
            sim = compute_cosine_similarity(embeddings, data.edge_index)

            # Make sure the dictionary is initialized for each layer_idx
            if layer_idx not in epoch_layer_similarities:
                epoch_layer_similarities[layer_idx] = [0.0, 0.0, 0.0]

            # Accumulate similarities
            epoch_layer_similarities[layer_idx][0] += sim[0]  # neighbor similarity
            epoch_layer_similarities[layer_idx][1] += sim[1]  # non-neighbor similarity
            epoch_layer_similarities[layer_idx][2] += sim[2]  # all-pair similarity

        num_val_batches += 1

    # Calculate average cosine similarity per layer for this epoch
    avg_neighbor_sim = {}
    avg_non_neighbor_sim = {}
    avg_all_pair_sim = {}

    for layer_idx, total_similarity in epoch_layer_similarities.items():
        avg_neighbor_sim[layer_idx] = total_similarity[0] / num_val_batches
        avg_non_neighbor_sim[layer_idx] = total_similarity[1] / num_val_batches
        avg_all_pair_sim[layer_idx] = total_similarity[2] / num_val_batches

        print(f"[E{epoch} L{layer_idx}] Neighbor: {avg_neighbor_sim[layer_idx]:.4f} | "
              f"Non-neigh: {avg_non_neighbor_sim[layer_idx]:.4f} | "
              f"All-pair: {avg_all_pair_sim[layer_idx]:.4f}")

    # Store the calculated similarities for the current epoch
    all_epoch_similarities[f'Epoch_{epoch}_Neighbor'] = avg_neighbor_sim
    all_epoch_similarities[f'Epoch_{epoch}_NonNeighbor'] = avg_non_neighbor_sim
    all_epoch_similarities[f'Epoch_{epoch}_AllPair'] = avg_all_pair_sim


    metrics.reset_parameters('validation',epoch)
    scheduler.step()
    end_time         = time.time()
    e_time           = end_time-start_time
    metrics.save_time(e_time)
    
    # EARLY-STOPPING
    early_stopping(metrics.valid_loss2[epoch], net)
    flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
    if early_stopping.FLAG == True:    estop_val = flag_value
    else:
        estop_val        = '@best: saving model...'; best_epoch = epoch+1
    output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

    if early_stopping.early_stop:
        print("> Early stopping")
        break
# SAVING MODEL
print(f"> DONE TRAINING !")
shutil.copy2('TRAINED/crystal-checkpoint.pt', f'TRAINED/{crystal_property}_{model_name}.pt')
print("Similarity keys:", all_epoch_similarities.keys())

# Save all cosine similarities to an Excel file
if all_epoch_similarities:  # Check if any similarities were calculated
    print("\nSaving all calculated cosine similarities...")
    final_df = pd.DataFrame.from_dict(all_epoch_similarities, orient='index')
    final_df.index.name = 'Epoch_Layer'
    final_df.to_excel("all_epochs_cosine_similarity.xlsx")
    print("Saved to all_epochs_cosine_similarity.xlsx") 
