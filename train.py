import os 
import os.path as op 
from utils import * 
from torch_geometric.loader import DataLoader

NUM_GEN = 30000 
BATCH_GEN = 500 # If the user run into issue with memory, try to reduce this number

if __name__ == '__main__' : 
    # set_seed(910)

    config = get_train_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    train_set, vocab_smi, vocab_graph, max_token = get_dataset(config.path_train_raw_file, config.path_train_processed_folder)
    
    update_config(config, {'vocab_smi': vocab_smi, 'vocab_graph': vocab_graph, 'max_token': max_token})

    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True)

    model, optim, annealer = get_model(config, device)

    for e in range(config.trained_epoch + 1, config.epoch + 1) : 
        model.train() 
        for i, data in enumerate(tqdm(train_loader, desc=f'Training epoch {e}')) : 
            inp_graph, inp_smi, inp_smi_mask, tgt_smi = convert_data(data, vocab_smi, device=device) 
            output = model(inp_graph, inp_smi, inp_smi_mask)
            loss = loss_fn(output, tgt_smi, annealer[e-1], config)

            loss.backward()
            if config.gradient_clipping : torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optim.step(), optim.zero_grad()

        if e % config.save_every == 0 and e >= config.start_save : checkpoint(model, optim, e, config)
        if e % config.generate_every == 0 and e >= config.start_generate: generate_molecule(model, config, NUM_GEN, op.join(config.path_generate_folder, f'epoch_{e}.txt'), batch=BATCH_GEN)