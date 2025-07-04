import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
SIZE = 128
BATCH_SIZE = 16
# EPOCH_LR = [(30,0.01),(30,0.001)]
EPOCH_LR = [(30, 0.01), (30, 0.001), (50, 0.001)]
CHECKPOINT = "/data/image_search"
DATA_FOLDER = "/data/pubfig_faces"
