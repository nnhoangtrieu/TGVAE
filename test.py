# with open('data/moses_train.txt', 'r') as f : 
#     contents = f.readlines() 
#     # contents = [c.strip() for c in contents if len(c) <=  and len(c) > 43]
#     contents = [c.strip() for c in contents]
#     contents = [c for c in contents if len(c) <= 57 and len(c) >= 40]

#     maxlen = len(max(contents, key=len))
    
#     print(len(contents))
#     print(maxlen)



import torch 
from torch.utils.data import Dataset, DataLoader 



class GenSet(Dataset) : 
    def __init__(self, d_latent) : 
        self.d_latent = d_latent
        

    def __len__(self) : 
        return 30000
    
    def __getitem__(self, idx) : 
        z = torch.randn(self.d_latent)
        tgt = torch.zeros(1, dtype=torch.long)
        return z, tgt 
    

dataset = GenSet(256)
train_loader = DataLoader(dataset, batch_size = 500)


for i, j in train_loader :
    print(i.shape, j.shape)