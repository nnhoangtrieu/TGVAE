# TGVAE: Transformer Graph Variational Autoencoder for Generative Molecular Design 

## Installation 
- Python 3.7+

```bash
git clone https://github.com/nnhoangtrieu/TGVAE.git
pip install requirements.txt
```

--- 
## Usage 

### Training 
```bash
python train.py 
```
### Arguments
This section describes the various hyperparameters used in the training code. They are grouped into model, training, loss function, and miscellaneous hyperparameters. Each parameter can be set via command-line arguments, allowing you to fine-tune the model architecture and training process.

#### Model Hyperparameters

* `-tr/--train`: Path to the training data file. Default: `moses_train.txt`
* `-de/--dim_encoder`: Dimension of the encoder's output representation. Default: `512`
* `-dd/--dim_decoder`: Dimension of the decoder's output representation. Default: `512`
* `-dl/--dim_latent`: Dimension of the latent space representation (often used in variational models). Default: `256`
* `-def/--dim_encoder_ff`: Dimension of the encoder feed-forward layer. Default: `512`
* `-ddf/--dim_decoder_ff`: Dimension of the decoder feed-forward layer. Default: `512`
* `-nel/--num_encoder_layer`: Number of layers in the encoder. Default: `4`
* `-ndl/--num_decoder_layer`: Number of layers in the decoder. Default: `4`
* `-neh/--num_encoder_head`: Number of attention heads in the encoder. Default: `1`
* `-ndh/--num_decoder_head`: Number of attention heads in the decoder. Default: `16`
* `-doe/--dropout_encoder`: Dropout rate applied in the encoder to prevent overfitting. Default: `0.3`
* `-dog/--dropout_gat`: Dropout rate for the graph attention mechanism (if applicable). Default: `0.3`
* `-dod/--dropout_decoder`: Dropout rate applied in the decoder to prevent overfitting. Default: `0.3`

#### Training Hyperparameters

* `-b/--batch`: Batch size used during training. Default: `128`
* `-e/--epoch`: Total number of training epochs. Default: `40`
* `-gc/--gradient_clipping`: Maximum allowed value for gradient clipping. Helps in preventing exploding gradients. Default: `5.0`

#### Loss Function Hyperparameters

* `-lkl/--loss_kl`: Specifies the type of KL-divergence loss computation (e.g., `mean` or other reduction methods). Default: `mean`
* `-lr/--learning_rate`: Learning rate for the optimizer. Default: `5e-4`
* `-wd/--weight_decay`: Weight decay (L2 regularization) factor applied during training. Default: `1e-6`
* `-aes/--anneal_epoch_start`: Epoch from which annealing of certain parameters (like KL weight) starts. Default: `0`
* `-aws/--anneal_weight_start`: Initial annealing weight for the loss function. Default: `0.00005`
* `-awe/--anneal_weight_end`: Final annealing weight target. Default: `1.0`

#### Other Hyperparameters

* `-se/--save_every`: Frequency (in epochs) at which the model is saved. Default: `1`
* `-ge/--generate_every`: Frequency (in epochs) at which sample generations are performed. Default: `1`
* `-ss/--start_save`: Epoch from which model saving starts. Default: `5`
* `-sg/--start_generate`: Epoch from which sample generation starts. Default: `5`
* `-n/--name`: Name or identifier for the experiment. Useful for logging and saving checkpoints. Default: `experiment_1`

---


### Generate molecules
```bash
python generate.py 
```