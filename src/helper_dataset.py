import gc
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import idx2numpy
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

#### Prepare dataset
class PrepareDataset:
    def __init__(self, config):
        self.config = config
        self.load_images()
        
    def load_images(self):
        data_dir = self.config.data_dir
        self.train_images = idx2numpy.convert_from_file(f'{data_dir}/train-images.idx3-ubyte')
        self.train_labels = idx2numpy.convert_from_file(f'{data_dir}/train-labels.idx1-ubyte')

        self.test_images = idx2numpy.convert_from_file(f'{data_dir}/t10k-images.idx3-ubyte')
        self.test_labels = idx2numpy.convert_from_file(f'{data_dir}/t10k-labels.idx1-ubyte')
        
        if not os.path.exists(os.path.join(data_dir, 'train_folds.csv')):
            # Make train labels into a dataframe to split into folds
            df = pd.DataFrame(self.train_labels, columns = ['label']).reset_index()
            df.rename(columns = {'index': 'id'}, inplace = True)
            self.df = self.split_dataset(df, seed = self.config.seed, save = True)
        else:
            self.df = pd.read_csv(os.path.join(data_dir, 'train_folds.csv'))
        
    # Load data and split into test train
    def split_dataset(self, df, seed, save = True):
        
        # shuffle
        df = df.sample(frac = 1, random_state = seed).reset_index(drop = True)
        df['kfold'] = -1
        
        # split into multiple folds
        kf = StratifiedKFold(n_splits=5)

        # populate the fold column
        y = df['label']
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_idx, 'kfold'] = fold

        # save the dataset
        if save:
            df.to_csv(f'{self.config.data_dir}/train_folds.csv')
            
        return df
    
    def make(self, fold):
        train = self.df[self.df['kfold'] != fold]
        valid = self.df[self.df['kfold'] == fold]
        
        self.train_ids = train['id'].values.tolist()
        self.valid_ids = valid['id'].values.tolist()
        
        self.X_train = self.train_images[self.train_ids, :, :]
        self.y_train = train['label'].values.tolist()
        
        self.X_valid = self.train_images[self.valid_ids, :, :]
        self.y_valid = valid['label'].values.tolist()
        
        self.test_ids = list(range(0, len(self.test_images)))
        self.X_test = self.test_images
        self.y_test = self.test_labels
        
### DATASET
class MNISTDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = X
        self.y = y
        self.ids = ids
        
        # normalize
        self.X = ((self.X / 255.) - 0.5) * 2        # scales to [-1, 1]
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        i = self.ids[idx]
        x = torch.tensor(self.X[idx, :, :], dtype=torch.float32)       # normalize
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return i, x, y
    
### DATALOADER
def get_dataloaders(config, fold):
    prep = PrepareDataset(config)
    prep.make(fold = fold)
    
    train = MNISTDataset(X = prep.X_train,  y = prep.y_train,   ids=prep.train_ids)
    valid = MNISTDataset(X = prep.X_valid,  y = prep.y_valid,   ids = prep.valid_ids)
    test  = MNISTDataset(X = prep.X_test,   y = prep.y_test,    ids = prep.test_ids)
    
    train_loader = DataLoader(dataset = train, 
                              batch_size = config.train_batch_size, 
                              shuffle=True, 
                              drop_last=False)
    
    valid_loader = DataLoader(dataset = valid, 
                              batch_size = config.valid_batch_size, 
                              shuffle=False)
    
    test_loader = DataLoader(dataset = test, 
                             batch_size = config.valid_batch_size, 
                             shuffle=True)
    
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    config.data_dir = 'inputs/mnist'
    
    _ = get_dataloaders(config, fold = 0)