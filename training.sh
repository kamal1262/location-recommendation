#!/usr/bin/env bash

# SM_MODEL_DIR=models/output/ SM_OUTPUT_DATA_DIR=data/output/ SM_CHANNEL_TRAIN=data/list_seq.p SM_CHANNEL_VOCAB=models/loc2name.p python \
#     -m src.ml.train_skipgram \
#     --batch-size=16 \
#     --embedding-dims=128 \
#     --initial-lr=0.025 \
#     --epochs=25 \
#     --shuffle=True
SM_MODEL_DIR=models/output/ SM_OUTPUT_DATA_DIR=data/output/ SM_CHANNEL_TRAIN=data/list_seq.p SM_CHANNEL_VOCAB=models/loc2name.p python -m src.sm.train_skipgram
