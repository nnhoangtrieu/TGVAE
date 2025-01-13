import os.path as op
from model.main import TGVAE
from utils import * 
from argparse import Namespace



if __name__ == '__main__' : 
    arg = get_generate_arg()

    path_config, path_snapshot, path_output = get_generate_path(arg)

    config = Namespace(**load(path_config))
    snapshot = load(path_snapshot)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = get_model(config, device, snapshot)

    for _ in tqdm(range(arg.num_gen // arg.batch), desc='Generating') : 
        smi_token = model.generate(num_gen=arg.batch, config=config)
        smi = convert_token(smi_token, config.vocab_smi)
        save(smi, path_output, mode='a')


