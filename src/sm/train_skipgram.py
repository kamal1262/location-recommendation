import os
import argparse
import sagemaker

import boto3
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.session import s3_input, Session
from sagemaker.amazon.amazon_estimator import get_image_uri 


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--vocab', type=str, default=os.environ['SM_CHANNEL_VOCAB'])

    args, _ = parser.parse_known_args()
    
    # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
    pytorch_estimator = PyTorch('src/ml/train_skipgram.py',
                            source_dir='.',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            role="arn:aws:iam::793999821937:role/SagemakerNotebookRole",
                            framework_version='1.6.0',
                            py_version='py3',
                            hyperparameters = {'epochs': 25, 'batch-size': 16, 'embedding-dims': 128, 'initial-lr': 0.025, 'shuffle': True})
    
    pytorch_estimator.fit({
        'train': args.train, 
        'vocab': args.vocab,
        })