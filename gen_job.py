scripts = ['python train.py --d_model 512 --d_latent 512 --d_ff 1024 --save_name "dimension"',
'python train.py --d_model 512 --d_latent 256 --d_ff 1024 --save_name "dimension"',
'python train.py --d_model 256 --d_latent 256 --d_ff 512 --save_name "dimension"',
'python train.py --d_model 256 --d_latent 128 --d_ff 512 --save_name "dimension"',
'python train.py --kl_w_start 0.0005 --kl_w_end 0.005 --save_name "kl_weight"',
'python train.py --kl_w_start 0.0005 --kl_w_end 0.01 --save_name "kl_weight"',
'python train.py --kl_w_start 0.0005 --kl_w_end 0.05 --save_name "kl_weight"',
'python train.py --kl_w_start 0.001 --kl_w_end 0.005 --save_name "kl_weight"',
'python train.py --kl_w_start 0.001 --kl_w_end 0.01 --save_name "kl_weight"',
'python train.py --kl_w_start 0.001 --kl_w_end 0.05 --save_name "kl_weight"',
'python train.py --n_epochs 40 --kl_w_start 0.0005 --kl_w_end 0.005 --save_name "kl_weight long epoch"',
'python train.py --n_epochs 40 --kl_w_start 0.0005 --kl_w_end 0.01 --save_name "kl_weight long epoch"',
'python train.py --n_epochs 40 --kl_w_start 0.0005 --kl_w_end 0.05 --save_name "kl_weight long epoch"',
'python train.py --n_epochs 40 --kl_w_start 0.001 --kl_w_end 0.005 --save_name "kl_weight long epoch"',
'python train.py --n_epochs 40 --kl_w_start 0.001 --kl_w_end 0.01 --save_name "kl_weight long epoch"',
'python train.py --n_epochs 40 --kl_w_start 0.001 --kl_w_end 0.05 --save_name "kl_weight long epoch"'
]


for i in range(len(scripts)) : 
    with open(f'job{i}.sub', 'w') as f : 
        f.write(f'''#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 3-0
#SBATCH -c 4
#SBATCH --mem 40g
#SBATCH --output graphvae-%j.out
#SBATCH --gres=gpu:a30:1
                
ml Python 
{scripts[i]}
 ''')
        

# for file in /home/80027464/graphvae/job*; do [ -f "$file" ] && sbatch "$file"; done


