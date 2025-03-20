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

- **`--train` (`-tr`)**  
  *Type:* `str`  
  *Default:* `moses_train.txt`  
  *Description:* Path to the training data file.

- **`--dim_encoder` (`-de`)**  
  *Type:* `int`  
  *Default:* `512`  
  *Description:* Dimension of the encoder's output representation.

- **`--dim_decoder` (`-dd`)**  
  *Type:* `int`  
  *Default:* `512`  
  *Description:* Dimension of the decoder's output representation.

- **`--dim_latent` (`-dl`)**  
  *Type:* `int`  
  *Default:* `256`  
  *Description:* Dimension of the latent space representation (often used in variational models).

- **`--dim_encoder_ff` (`-def`)**  
  *Type:* `int`  
  *Default:* `512`  
  *Description:* Dimension of the encoder feed-forward layer.

- **`--dim_decoder_ff` (`-ddf`)**  
  *Type:* `int`  
  *Default:* `512`  
  *Description:* Dimension of the decoder feed-forward layer.

- **`--num_encoder_layer` (`-nel`)**  
  *Type:* `int`  
  *Default:* `4`  
  *Description:* Number of layers in the encoder.

- **`--num_decoder_layer` (`-ndl`)**  
  *Type:* `int`  
  *Default:* `4`  
  *Description:* Number of layers in the decoder.

- **`--num_encoder_head` (`-neh`)**  
  *Type:* `int`  
  *Default:* `1`  
  *Description:* Number of attention heads in the encoder.

- **`--num_decoder_head` (`-ndh`)**  
  *Type:* `int`  
  *Default:* `16`  
  *Description:* Number of attention heads in the decoder.

- **`--dropout_encoder` (`-doe`)**  
  *Type:* `float`  
  *Default:* `0.3`  
  *Description:* Dropout rate applied in the encoder to prevent overfitting.

- **`--dropout_gat` (`-dog`)**  
  *Type:* `float`  
  *Default:* `0.3`  
  *Description:* Dropout rate for the graph attention mechanism (if applicable).

- **`--dropout_decoder` (`-dod`)**  
  *Type:* `float`  
  *Default:* `0.3`  
  *Description:* Dropout rate applied in the decoder to prevent overfitting.

#### Training Hyperparameters

- **`--batch` (`-b`)**  
  *Type:* `int`  
  *Default:* `128`  
  *Description:* Batch size used during training.

- **`--epoch` (`-e`)**  
  *Type:* `int`  
  *Default:* `40`  
  *Description:* Total number of training epochs.

- **`--gradient_clipping` (`-gc`)**  
  *Type:* `float`  
  *Default:* `5.0`  
  *Description:* Maximum allowed value for gradient clipping. Helps in preventing exploding gradients.

#### Loss Function Hyperparameters

- **`--loss_kl` (`-lkl`)**  
  *Type:* `str`  
  *Default:* `mean`  
  *Description:* Specifies the type of KL-divergence loss computation (e.g., `mean` or other reduction methods).

- **`--learning_rate` (`-lr`)**  
  *Type:* `float`  
  *Default:* `5e-4`  
  *Description:* Learning rate for the optimizer.

- **`--weight_decay` (`-wd`)**  
  *Type:* `float`  
  *Default:* `1e-6`  
  *Description:* Weight decay (L2 regularization) factor applied during training.

- **`--anneal_epoch_start` (`-aes`)**  
  *Type:* `int`  
  *Default:* `0`  
  *Description:* Epoch from which annealing of certain parameters (like KL weight) starts.

- **`--anneal_weight_start` (`-aws`)**  
  *Type:* `float`  
  *Default:* `0.00005`  
  *Description:* Initial annealing weight for the loss function.

- **`--anneal_weight_end` (`-awe`)**  
  *Type:* `float`  
  *Default:* `1.0`  
  *Description:* Final annealing weight target.

#### Other Hyperparameters

- **`--save_every` (`-se`)**  
  *Type:* `int`  
  *Default:* `1`  
  *Description:* Frequency (in epochs) at which the model is saved.

- **`--generate_every` (`-ge`)**  
  *Type:* `int`  
  *Default:* `1`  
  *Description:* Frequency (in epochs) at which sample generations are performed.

- **`--start_save` (`-ss`)**  
  *Type:* `int`  
  *Default:* `5`  
  *Description:* Epoch from which model saving starts.

- **`--start_generate` (`-sg`)**  
  *Type:* `int`  
  *Default:* `5`  
  *Description:* Epoch from which sample generation starts.

- **`--name` (`-n`)**  
  *Type:* `str`  
  *Default:* `experiment_1`  
  *Description:* Name or identifier for the experiment. Useful for logging and saving checkpoints.

---


### Generate molecules
```bash
python generate.py 
```