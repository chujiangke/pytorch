from torchvision import transforms
transforms_list = [
    # 图像尺寸转为224x128
    transforms.Resize((224, 128)),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 补边
    transforms.Pad(padding=(10, 20, 10, 20), fill=0, padding_mode='constant'),
    # 随机裁剪改为 256*128
    transforms.RandomCrop((256, 128)),
    #转为张量类型
    transforms.ToTensor(),
    #标准化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
transforms_demo = transforms.Compose(transforms_list)
 