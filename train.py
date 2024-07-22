import os 
import torch 
import argparse
from tqdm import tqdm 
from model.main import TGVAE
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper.utils import preprocess, get_mask, loss_fn, monotonic_annealer, load_file, save_file, seed_torch, read_smi_file, MyData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
parser = argparse.ArgumentParser() 
parser.add_argument('-tr', '--train', type=str, default='moses-train')
parser.add_argument('-te', '--test', type=str, default='moses-test')
parser.add_argument('-dm', '--d_model', nargs=2, type=int, default=[512, 512])
parser.add_argument('-df', '--d_ff', type=int, default=1024)
parser.add_argument('-nl', '--n_layer', nargs=2, type=int, default=[8, 8])
parser.add_argument('-nh', '--n_head', nargs=2, type=int, default=[1, 16])
parser.add_argument('-do', '--dropout', type=float, default=0.5)
parser.add_argument('-b', '--batch', type=int, default=128)
parser.add_argument('-ed', '--edge_dim', type=int, default=None)
parser.add_argument('-em', '--encoder_mode', type=str, default='none')
parser.add_argument('-pm', '--pool_mode', type=str, default='add')
parser.add_argument('-gm', '--gnn_mode', type=str, default='res+')
parser.add_argument('-gc', '--clip', type=int, default=5)
parser.add_argument('-n', '--name', type=str, default='experiment_1')
arg = parser.parse_args() 


seed_torch()

raw_folder = 'data/raw'
processed_folder = 'data/processed'
checkpoint_folder = f'output/checkpoint/{arg.name}'
tensorboard_folder = f'output/tensorboard_train/{arg.name}'

config = load_file(f'{checkpoint_folder}/config.json') if os.path.exists(f'{checkpoint_folder}/config.json') else None

if not config : 
    epoch = 0 
    config = vars(arg)
    config['epoch'] = epoch
    os.makedirs(checkpoint_folder, exist_ok=True)
    save_file(config, f'{checkpoint_folder}/config.json')
else : 
    epoch = config['epoch']
    print(f'Last epoch: {epoch}')


if not os.path.exists(f'{processed_folder}/{config["train"]}') : 
    tr_smi_list = read_smi_file(f'{raw_folder}/{config["train"]}.txt')
    tr_nf, tr_ei, tr_ew, tr_token, tr_smivocab, tr_nodevocab, tr_edgevocab, tr_maxlen = preprocess(tr_smi_list, f'{processed_folder}/{config["train"]}')

if not os.path.exists(f'{processed_folder}/{config["test"]}') : 
    te_smi_list = read_smi_file(f'{raw_folder}/{config["test"]}.txt')
    te_nf, te_ei, te_ew, te_token, te_smivocab, te_nodevocab, te_edgevocab, te_maxlen = preprocess(te_smi_list, f'{processed_folder}/{config["test"]}')

else : 
    files = ['nf_list.pt', 'ei_list.pt', 'ew_list.pt', 'token_list.pt', 'max_len.pt', 'smi_vocab.json', 'node_vocab.json', 'edge_vocab.json']
    tr_nf, tr_ei, tr_ew, tr_token, tr_maxlen, tr_smivocab, tr_nodevocab, tr_edgevocab = [load_file(f'{processed_folder}/{config["train"]}/{file}') for file in tqdm(files, desc=f'Loading {config["train"]} data')]
    te_nf, te_ei, te_ew, te_token, te_maxlen, te_smivocab, te_nodevocab, te_edgevocab = [load_file(f'{processed_folder}/{config["test"]}/{file}') for file in tqdm(files, desc=f'Loading {config["test"]} data')]



train_set = [MyData(x=nf, edge_index=ei, edge_attr=ew, smi=smi) for nf, ei, ew, smi in zip(tr_nf, tr_ei, tr_ew, tr_token)]
test_set = [MyData(x=nf, edge_index=ei, edge_attr=ew, smi=smi) for nf, ei, ew, smi in zip(te_nf, te_ei, te_ew, te_token)]
train_loader = DataLoader(train_set, batch_size=arg.batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=arg.batch, shuffle=False)




model = TGVAE(d_model=config["d_model"],
                d_ff=config["d_ff"],
                edge_dim=config["edge_dim"],
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                dropout=config["dropout"],
                encoder_mode=config["encoder_mode"],
                pool_mode=config["pool_mode"],
                gnn_mode=config["gnn_mode"],
                smi_vocab_size=len(tr_smivocab),
                node_vocab_size=len(tr_nodevocab),
                edge_vocab_size=len(tr_edgevocab)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)    
annealer = monotonic_annealer(100, 0, 0.00005, 0.0001)
writer = SummaryWriter(f'{tensorboard_folder}/{arg.name}')


if epoch != 0 :
    model.load_state_dict(load_file(f'{checkpoint_folder}/snapshot_{epoch}.pt')['MODEL_STATE'])
    optim.load_state_dict(load_file(f'{checkpoint_folder}/snapshot_{epoch}.pt')['OPTIMIZER_STATE'])
    print(f'Loaded model and optimizer from epoch {epoch}')



for e in range(epoch + 1, 101) : 
    train_loss, train_recon_loss, train_kl_loss, test_loss, test_recon_loss, test_kl_loss = 0, 0, 0, 0, 0, 0
    beta = annealer[e-1]

    model.train() 
    for data in tqdm(train_loader, desc=f'Training epoch {e}') : 
        graph = data.to(device) 
        smi = graph.clone().smi.to(device) 
        smi_mask = get_mask(smi[:, :-1], tr_smivocab)
        pred, mu, sigma = model(graph, smi[:, :-1], smi_mask)
        loss, recon_loss, kl_loss = loss_fn(pred, smi[:, 1:], mu, sigma, beta, tr_smivocab, arg.batch)
        loss.backward(), optim.step(), optim.zero_grad()
        if arg.clip: 
            clip_grad_norm_(model.parameters(), arg.clip) 
        train_loss, train_recon_loss, train_kl_loss = [train_loss + loss.item(), train_recon_loss + recon_loss.item(), train_kl_loss + kl_loss.item()]

    model.eval()
    for data in tqdm(test_loader, desc=f'Testing epoch {e}') : 
        graph = data.to(device) 
        smi = graph.clone().smi.to(device) 
        smi_mask = get_mask(smi[:, :-1], te_smivocab)
        pred, mu, sigma = model(graph, smi[:, :-1], smi_mask)
        loss, recon_loss, kl_loss = loss_fn(pred, smi[:, 1:], mu, sigma, beta, te_smivocab, arg.batch)
        test_loss, test_recon_loss, test_kl_loss = [test_loss + loss.item(), test_recon_loss + recon_loss.item(), test_kl_loss + kl_loss.item()]

    
    
    writer.add_scalar('loss/train', train_loss/len(train_loader), e), writer.add_scalar('loss/test', test_loss/len(test_loader), e), writer.add_scalar('kl/train', train_kl_loss/len(train_loader), e), writer.add_scalar('kl/test', test_kl_loss/len(test_loader), e), writer.add_scalar('recon/train', train_recon_loss/len(train_loader), e), writer.add_scalar('recon/test', test_recon_loss/len(test_loader), e)

    snapshot = {
        'MODEL_STATE': model.state_dict(),
        'OPTIMIZER_STATE': optim.state_dict()
    }
    config["epoch"] = e
    save_file(snapshot, f'output/checkpoint/{arg.name}/snapshot_{e}.pt'), save_file(config, f'output/checkpoint/{arg.name}/config.json')
    
