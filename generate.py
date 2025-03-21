import os.path as op
from model.main import TGVAE
from utils import * 
from argparse import Namespace


if __name__ == '__main__' : 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-s', '--snapshot', type=int, default=0)
    parser.add_argument('-pcf', '--path_config', type=str, default=None)
    parser.add_argument('-pss', '--path_snapshot', type=str, default=None)
    parser.add_argument('-ng', '--num_gen', type=int, default=30000)
    parser.add_argument('-b', '--batch', type=int, default=500)
    parser.add_argument('-o', '--output', type=str, default=None)
    arg = parser.parse_args()

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    if op.isdir(arg.output) : 
        output = op.join(arg.output, f'{time}.txt')
    else : 
        if arg.output.endswith('.txt') : 
            output = arg.output 
        else : 
            output = time + '.txt'

    path_script = op.dirname(op.abspath(__file__))
    path_checkpoint = op.join(path_script, 'checkpoint', arg.name)
    path_config = arg.path_config if arg.path_config else op.join(path_checkpoint, 'config.json')
    path_snapshot = arg.path_snapshot if arg.path_snapshot else op.join(path_checkpoint, f'snapshot_{arg.snapshot}.pt') 

    config = Namespace(**load(path_config))
    snapshot = load(path_snapshot)
    model = get_model(config, device, generate_snapshot=snapshot)

    for _ in tqdm(range(arg.num_gen // arg.batch), desc='Generating') : 
        smi_token = model.generate(config, arg.batch)
        smi = convert_token(smi_token, config.vocab_smi)
        save(smi, arg.output, mode='a')

