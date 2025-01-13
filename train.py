import os 
import os.path as op 
from utils import * 
from torch_geometric.loader import DataLoader


if __name__ == '__main__' : 
    set_seed(910)

    config = get_train_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    train_set, vocab_smi, vocab_graph, max_token = get_dataset(config.path_train_raw_file, config.path_train_processed_folder)
    test_set, _, _, _ = get_dataset(config.path_test_raw_file, config.path_test_processed_folder)
    
    os.makedirs(config.path_checkpoint_folder, exist_ok=True)
    update_config(config, {'vocab_smi': vocab_smi,
                           'vocab_graph': vocab_graph,
                           'max_token': max_token})

    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch, shuffle=False)

    model, optim, annealer = get_model(config, device)

    for e in range(config.trained_epoch + 1, config.epoch + 1) : 
        train_loss, test_loss = 0, 0

        model.train() 
        for data in tqdm(train_loader, desc=f'Training epoch {e}') : 
            inp_graph, inp_smi, inp_smi_mask, tgt_smi = convert_data(data, vocab_smi, device=device) 
            output = model(inp_graph, inp_smi, inp_smi_mask)
            loss = loss_fn(output, tgt_smi, config)
            train_loss += loss.item()
            loss.backward(), optim.step(), optim.zero_grad()
            
        checkpoint(model, optim, e, config)

        model.eval()
        with torch.no_grad() : 
            for data in tqdm(test_loader, desc=f'Testing epoch {e}') : 
                inp_graph, inp_smi, inp_smi_mask, tgt_smi = convert_data(data, vocab_smi, device=device)
                output = model(inp_graph, inp_smi, inp_smi_mask)
                loss = loss_fn(output, tgt_smi, config)
                test_loss += loss.item() 

        print(f'Epoch {e} - Train loss: {train_loss/len(train_loader):.2f} / Test loss: {test_loss/len(test_loader):.2f}')
