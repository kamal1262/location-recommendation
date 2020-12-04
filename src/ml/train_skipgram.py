import torch
import numpy as np

from torch import nn
from torch import optim

import torch.functional as F
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import gzip
import pickle
import argparse
import datetime
import itertools

from tqdm import tqdm
from typing import Any
from collections import Counter
from typing import Dict, List, Tuple

from src.config import MODEL_PATH
from src.utils.logger import logger
from src.ml.skipgram import SkipGram
from src.utils.io_utils import save_model
from src.ml.data_loader import Sequences, SequencesDataset

from scipy.spatial import distance

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training REA IPP MY location embeddings on Pytorch')
    # parser.add_argument('--read_seq', type=str, help='Path to sequences.p')
    # parser.add_argument('--read_loc_dict', type=str, help='Path to location dict for vocabs')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--batch-size', type=int, help='Batchsize for dataloader', default=16)
    parser.add_argument('--embedding-dims', type=int, help='Embedding size', default=128)
    parser.add_argument('--initial-lr', type=float, help='Initial LR', default=0.025)
    parser.add_argument('--epochs', type=int, help='No of Epochs', default=25)
    parser.add_argument('--shuffle', type=bool, help='Shuffle ?', default=True)
    parser.add_argument('--output-data-dir', type=str, help='Sagemaker output dir', default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, help='Sagemaker training dataset', default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--vocab', type=str, help='Sagemaker vocab', default=os.environ['SM_CHANNEL_VOCAB'])
    # parser.add_argument('--test', type=str, help='Sagemaker test dataset', help='Sagemaker output dir', default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    shuffle = args.shuffle
    embedding_dims = args.embedding_dims
    epochs = args.epochs
    initial_lr = args.initial_lr
    batch_size = args.batch_size
    model_dir = args.model_dir
    
    read_seq = args.train
    read_loc_dict = args.vocab
    
    with open(f"{read_seq}/list_seq.p", 'rb') as f:
        list_seq = pickle.load(f)
        
    with open(f"{read_loc_dict}/loc2name.p", 'rb') as f:
        dict_loc = pickle.load(f)
    
    vocab_size = len(dict_loc) # 14699
    
    # Tensorboard writer config
    writer = SummaryWriter('/opt/output/tensorboard/')
    
    # Load dataloader
    sequences = Sequences(seq_list=list_seq, vocab_dict=dict_loc)
    dataset = SequencesDataset(sequences)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=dataset.collate)

    # Initialize model
    skipgram = SkipGram(vocab_size, embedding_dims).to(device)

    # Train loop
    optimizer = optim.SparseAdam(list(skipgram.parameters()), lr=initial_lr)
    
    results = []
    start_time = datetime.datetime.now()
    for epoch in tqdm(range(epochs), total=epochs, position=0, leave=True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
        running_loss = 0
        
        # Training loop
        for i, batches in enumerate(dataloader):
            centers = batches[0].to(device)
            contexts = batches[1].to(device)
            neg_contexts = batches[2].to(device)

            optimizer.zero_grad()
            loss = skipgram.forward(centers, contexts, neg_contexts)
            loss.backward()
            optimizer.step()

            scheduler.step()
            running_loss = running_loss * 0.9 + loss.item() * 0.1
            
        logger.info("Epoch: {}, Loss: {:.4f}, Lr: {:.6f}".format(epoch, running_loss, optimizer.param_groups[0]['lr']))
        writer.add_scalar('Training loss', loss.item(), epoch)
        writer.add_scalar('Running loss', running_loss, epoch)

        results.append([epoch, i, running_loss])
        running_loss = 0

        # save model
        # current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        # state_dict_path = '{}/skipgram_epoch_{}_{}.pt'.format(MODEL_PATH, epoch, current_datetime)
        # torch.save(skipgram.state_dict(), state_dict_path)
        # logger.info('Model state dict saved to {}'.format(state_dict_path))

    end_time = datetime.datetime.now()
    time_diff = round((end_time - start_time).total_seconds() / 60, 2)
    logger.info('Total time taken: {:,} minutes'.format(time_diff))
    
    save_model(skipgram, model_dir)
    skipgram.save_embeddings(f"{model_dir}/embeddings.npy")