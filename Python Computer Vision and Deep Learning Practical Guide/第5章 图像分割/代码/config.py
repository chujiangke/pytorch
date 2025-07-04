import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_folder = "/data/object_detection_segment/segmentation"
data_folder = "/data/object_detection_segment/object_detection"
sr_data_folder = "/data/super_resolution"

# num_epoch = 100
batch_size = 8
lr = 0.001
epoch_lr = [(20, 0.01), (10, 0.001), (10, 0.0001)]

checkpoint = "/data/chapter_three/net.pth"
sr_checkpoint = "/data/chapter_three/sr.pth"
