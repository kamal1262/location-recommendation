python -m src.ml.train_skipgram --read_seq=models/list_seq.p --read_loc_dict=models/loc2name.p --batch_size=16 --embedding_dims=128 --initial_lr=0.025 --epochs=25 --shuffle=True
# python -m src.ml.train_skipgram --read_seq models/list_seq.p --read_loc_dict models/loc2name.p --batch_size 16 --embedding_dims 128 --initial_lr 0.025 --epochs 25 --shuffle True
# python -m src.ml.train_skipgram models/list_seq.p models/loc2name.p 16 128 0.025 25 True
