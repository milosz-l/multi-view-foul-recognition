import torchvision.transforms as transforms

def get_augmentation(data_aug: bool):
    if data_aug:
        print("Using data augmentation")
        transformAug = transforms.Compose([
                                        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                                        transforms.RandomAffine(degrees=(0, 0), translate=(0.2, 0.2), scale=(0.8, 1.2)),
                                        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                        transforms.RandomRotation(degrees=10),
                                        transforms.ColorJitter(brightness=0.6, saturation=0.6, contrast=0.6),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        ])
    else:
        print("Not using data augmentation")
        transformAug = None
    return transformAug

