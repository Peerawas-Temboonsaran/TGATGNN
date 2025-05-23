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
random_num      =  456;random.seed(random_num)

# MODEL HYPER-PARAMETERS
num_epochs      = 300
learning_rate   = 5e-3
batch_size      = 256

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]
test_param      = {'batch_size':256, 'shuffle': False}

# DATALOADER/ TARGET NORMALIZATION
src_CIF         = 'CIF-DATA_NEW' if data_src == 'NEW' else 'CIF-DATA'
dataset         = pd.read_csv(f'DATA/{src_CIF}/id_prop.csv',names=['material_ids','label']).sample(frac=1,random_state=random_num)
NORMALIZER      = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA    = CIF_Dataset(dataset, root_dir = f'DATA/{src_CIF}/',**RSM)
idx_list        = list(range(len(dataset)))
random.shuffle(idx_list)

train_idx,test_val = train_test_split(idx_list,train_size=training_num,random_state=random_num)
test_idx,_         = train_test_split(test_val,test_size=0.5,random_state=random_num)
testing_set        = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm_action, df=dataset,src=data_src)

# NEURAL-NETWORK
if model_name == 'TGATGNN-1': # ADDED: Model selection
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
    num_transformer_layers = 2
    net = net_class(n_heads, transformer_heads, classification, neurons=number_neurons, nl=number_layers,
             xtra_layers=xtra_l, global_attention=global_att, unpooling_technique=attention_technique,
             concat_comp=concat_comp, edge_format=data_src, num_transformer_layers=num_transformer_layers).to(device)
    
else:
    raise ValueError(f"Invalid model name: {model_name}")

# LOSS & OPTMIZER & SCHEDULER
if classification == 1: criterion   = nn.CrossEntropyLoss().cuda(); funct = torch_accuracy
else                  : criterion   = nn.SmoothL1Loss().cuda()    ; funct = torch_MAE
optimizer         = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-1)

# LOADING MODEL
net.interpretation = True 
# For AGAT comment line 98, uncomment line 99-101
net.load_state_dict(torch.load(f'TRAINED/{crystal_property}_{model_name}.pt',map_location=device))
#checkpoint = torch.load(f'TRAINED/{crystal_property}.pt', map_location=device)
#filtered_state_dict = {k: v for k, v in checkpoint.items() if "cluster_att" not in k and "comp_atten" not in k}
#net.load_state_dict(filtered_state_dict)

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct,device)

print(f'> EVALUATING MODEL ...')
# TESTING PHASE
test_loader    = torch_DataLoader(dataset=testing_set,    **test_param)
true_label, pred_label = torch.tensor([]).to(device),torch.tensor([]).to(device)
testset_idx    = torch.tensor([]).to(device)
num_elements   = torch.tensor([]).to(device)
net.eval()

for data in test_loader:
    data        = data.to(device)
    with torch.no_grad():
        predictions, _ = net(data)
    print(f'(batch --- :{data.y.shape[0]:4})','---',metrics.eval_func(predictions,data.y).item())
    true_label      = torch.cat([true_label,data.y.float()],dim=0)
    pred_label      = torch.cat([pred_label,predictions.float()],dim=0)
    testset_idx     = torch.cat([testset_idx,data.the_idx],dim=0)
    num_elements    = torch.cat([num_elements,data.num_atoms],dim=0)


test_result    = metrics.eval_func(pred_label,true_label)
print(f'RESULT ---> {test_result:.5f}')
true_label   = true_label.cpu().numpy()
pred_label   = pred_label.cpu().numpy()
testset_idx  = testset_idx.cpu().numpy()
num_elements = num_elements.cpu().numpy()

csv_file         =  pd.DataFrame(zip(dataset.iloc[testset_idx].material_ids.values,true_label,pred_label,num_elements,testset_idx),
    columns=['material_ids',f'Measured {crystal_property}',f'Predicted {crystal_property}','Num_nodes','General_id'])
csv_file.to_csv(f'RESULTS/{crystal_property}_results.csv')

