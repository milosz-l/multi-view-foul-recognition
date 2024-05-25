import torchvision.transforms as transforms

def get_augmentation(data_aug: bool):
    if data_aug:
        print("Using data augmentation")
        transformAug = transforms.Compose([
                                        transforms.RandomAffine(
                                            degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                        transforms.RandomPerspective(
                                            distortion_scale=0.3, p=0.5),
                                        transforms.RandomRotation(degrees=5),
                                        transforms.ColorJitter(
                                            brightness=0.5, saturation=0.5, contrast=0.5),
                                        transforms.RandomHorizontalFlip()
                                        ])
    else:
        print("Not using data augmentation")
        transformAug = None
    return transformAug

