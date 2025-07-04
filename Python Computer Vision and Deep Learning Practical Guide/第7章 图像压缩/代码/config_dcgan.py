import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
lr = 2e-4
size = 64
num_epoch = 100
noise_length = 100
# data_folder = "/data/super_resolution"
data_folder = "/data/pubfig_faces"
checkpoint_D = "/data/chapter_four/D.pth"
checkpoint_G = "/data/chapter_four/G.pth"
