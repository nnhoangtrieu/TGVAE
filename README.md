# TGVAE: Transformer Graph Variational Autoencoder for Drug Design

This repo contains the PyTorch and Pytorch Geometric implementation of VAE with Graph Attention Network and Transformer Architecture for drug design. The code is organized by folders that corresponding to the following sections: 
- **data**: contain SMILES dataset from ChemBL (1.4M)
- **genmol**: molecules generated from trained model via generate.py will be stored here
- **genmol-train**: molecues generated during training after epoch will be stored here
- **model**: different architecture of models
- **tensorboard**: results data for analysis

## Training
The default configuration of the model is:

**d_model**: 512 | **d_latent**: 256 | **d_ff**: 1024 | **num_head**: 8 | **num_layer**: 8 | **dropout**: 0.5 | **lr**: 0.0003 | **epochs**: 32 | **batch_size**: 128 | **max_len**: 30 | **kl_type**: monotonic | **kl_start**: 0 | **kl_w_start**: 0 | **kl_w_end**: 0.0003 | **kl_cycle**: 4 | **kl_ratio**: 0.9 | **name_checkpoint**: model | **epoch_checkpoint**: -1 

- The dimension of model (d_model) will be use throughout the Transformer Layer and  will be increased in the inner Feed Forward Layer (d_ff). Encoder will finally compress the the input to latent space (d_latent). 

You can retrain the model with the default configuration with a command

```bash
python train.py
```

The model is trained with SMILES that has length < 30 (max_len = 30) (~100k data). You can also retrain with default configuration but different maximum length by
```bash
python train.py --max_len 40 
```

## Generate Molecules
You can look at folder **gen_train** to look at the recorded validity, uniqueness, and novelty the generated molecules, but most importantly is to look at the generate molecules themselves to really judge the performance of the model at each epoch. You can then specify the checkpoint of the model at which epoch to generate molecules. The generated molecules will be save in folder **genmol**

```bash
python generate.py --name_checkpoint "your_name_checkpoint" --epoch_checkpoint 5
```