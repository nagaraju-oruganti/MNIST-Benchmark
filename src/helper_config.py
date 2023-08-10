import random, os
import numpy as np
import torch

### SEED EVERYTHING
def seed_everything(seed: int = 42):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f'set seed to {seed}')

class Config:
    
    # random seed
    seed = 3407
    seed_everything(seed = seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # repos
    data_dir = '../inputs/mnist'
    models_dir = '../models'
    
    # model name
    model_name = ''
    
    # Train parameters
    train_batch_size = 16
    valid_batch_size = 32
    