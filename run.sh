SM_MODEL_DIR=models/output/ SM_OUTPUT_DATA_DIR=models/output/ SM_CHANNEL_TRAIN=data/list_seq.p,models/loc2name.p python \
    -m src.ml.train_skipgram \
    --train=data/list_seq.p,models/loc2name.p \
    --model-dir=models/output/ --batch_size=16 \
    --embedding_dims=128 \
    --initial_lr=0.025 \
    --epochs=25 \
    --shuffle=True
