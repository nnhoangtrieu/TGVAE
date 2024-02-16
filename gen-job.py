scripts = ['python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 2 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 2 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 4 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 6 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 4 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_w_start 0.0001 --kl_w_end 0.01 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 2 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 2 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 4 --kl_ratio 1.0 --save_name "43up"',
'python train.py --max_len 57 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.01 --kl_cycle 4 --kl_ratio 1.0 --save_name "43up"'
]


for i in range(len(scripts)) : 
    with open(f'job{i}.sub', 'w') as f : 
        f.write(f'''#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 3-0
#SBATCH -c 4
#SBATCH --mem 40g
#SBATCH --output 43up/job-out/%j.out
#SBATCH --gres=gpu:a30:1
                
ml Python 
{scripts[i]}
 ''')
        

# for file in /home/80027464/graphvae/job*; do [ -f "$file" ] && sbatch "$file"; done


