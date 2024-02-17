import re 




scripts = [
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 4 --kl_ratio 1',
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 8 --kl_ratio 1',
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 10 --kl_ratio 1',
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.005 --kl_cycle 4 --kl_ratio 1',
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.005 --kl_cycle 8 --kl_ratio 1',
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.005 --kl_cycle 10 --kl_ratio 1',

'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "monotonic" --kl_w_start 0.0001 --kl_w_end 0.01'
'train-mulnode.py --max_len 100 --batch 1024 --n_layers 8 --lr 0.00025 --kl_type "monotonic" --kl_w_start 0.0001 --kl_w_end 0.05'
]


check = []

for i, script in enumerate(scripts) :
    split = script.split('--')[1:]
    batch = int(re.findall(f'--batch (\d+)',script)[0])
    print("Batch not divisible by 128") if batch % 128 != 0 else None
    split = [s.replace(' ','').replace('"','') for s in split]
    save_name = ' '.join(split)
    if save_name in check : 
        print('Name existed')
        exit()
    else :
        check.append(save_name)

    with open(f'job{i}.sub', 'w') as f : 
        f.write(f'''#!/bin/bash
#SBATCH --job-name=trieu-nguyen
#SBATCH --time 3-0
#SBATCH --nodes={batch // 128}
#SBATCH --ntasks={batch // 128}
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --output=job-out/{'|'.join(split)}.out

export MASTER_ADDR=$(scontrol show hostname ${{SLURM_NODELIST}} | head -n 1)

ml Python 
srun python -m torch.distributed.launch --use_env \
--nnodes {batch // 128} \
--nproc_per_node 1 \
--rdzv_id $RANDOM \ 
--rdzv_backend c10d \ 
--rdzv_endpoint $MASTER_ADDR:29500 \ 
{script} --save-name "{save_name}"
 ''')
        






# for file in /home/80027464/graphvae/job*; do [ -f "$file" ] && sbatch "$file"; done

