#!/usr/bin/env bash

# SM_MODEL_DIR=models/output/ SM_OUTPUT_DATA_DIR=data/output/ SM_CHANNEL_TRAIN=data/list_seq.p SM_CHANNEL_VOCAB=data/loc2name.p python \
#     -m src.ml.train_skipgram \
#     --batch-size=16 \
#     --embedding-dims=128 \
#     --initial-lr=0.025 \
#     --epochs=25 \
#     --shuffle=True

SM_MODEL_DIR=/opt/ml/output/model/ \
    SM_OUTPUT_DATA_DIR=/opt/ml/output/data/ \
    SM_CHANNEL_TRAIN=s3://location-recommendation/02120202-skipgram/data/list_seq.p \
    SM_CHANNEL_VOCAB=s3://location-recommendation/02120202-skipgram/data/loc2name.p \
    SM_CURRENT_HOST="skipgram-location" \
    SAGEMAKER_REGION="ap-southeast-1" \
    S3_PATH="s3://location-recommendation/02120202-skipgram/" \
    python -m src.sm.train_skipgram
