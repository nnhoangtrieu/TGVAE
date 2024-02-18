# import re

# scripts = [
# 'train.py --max_len 34 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 34 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 34 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 2 --kl_ratio 0.5',
# 'train.py --max_len 34 --batch 128 --n_layers 8 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 2 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 9 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 9 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 9 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 2 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 9 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 2 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 10 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 10 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 4 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 10 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.0005 --kl_cycle 2 --kl_ratio 0.5',
# 'train.py --max_len 30 --batch 128 --n_layers 10 --n_epochs 100 --kl_type "cyclic" --kl_w_start 0.0001 --kl_w_end 0.001 --kl_cycle 2 --kl_ratio 0.5'
# ]



# check = []

# for i, script in enumerate(scripts) :
#     split = script.split('--')[1:]
#     split = [s.replace(' ','').replace('"','').replace("'",'') for s in split]
#     save_name = '|'.join(split)
#     if save_name in check : 
#         print('Name existed')
#         exit()
#     else :
#         check.append(save_name)

#     with open(f'job{i}.sub', 'w') as f : 
#         f.write(f'''#!/bin/bash
                
# #SBATCH --job-name=trieu-nguyen
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:a30:1
# #SBATCH --output=job-out/single-gpu/{save_name}.out

# ml Python 
# python {script} --save_name "{save_name}"
#  ''')
        


#     with open(f'moses{i}.sub', 'w') as f : 
#         f.write(f'''#!/bin/bash
                
# #SBATCH --job-name=trieu-nguyen
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:a30:1
# #SBATCH --output=job-out/single-gpu/moses-{'|'.join(split)}.out

# ml Python 
# python get-metrics.py --save_name "{save_name}"
#  ''')




import os 

for i, save_name in enumerate(os.listdir('checkpoint/single-gpu')) : 
    with open(f'moses{i}.sub', 'w') as f : 
        f.write(f'''#!/bin/bash
                
#SBATCH --job-name=trieu-nguyen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --output=job-out/single-gpu/moses-{save_name}.out

ml Python 
python get-metrics.py --save_name "{save_name}"
 ''')




# # for file in /home/80027464/graphvae/job*; do [ -f "$file" ] && sbatch "$file"; done
# # for file in /home/80027464/graphvae/moses*; do [ -f "$file" ] && sbatch "$file"; done

