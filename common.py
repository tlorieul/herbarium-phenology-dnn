from torchvision import transforms


def get_data_transformation(keep_image_ratio, downsample_image):
    train_transform = transforms.Compose([
        transforms.RandomRotation(45)
    ])
    test_transform = transforms.Compose([])

    if keep_image_ratio:
        if downsample_image:
            import PIL

            def downsampling(img):
                new_size = (img.size[0]/2, img.size[1]/2)
                return img.resize(size=new_size, resample=PIL.Image.BILINEAR)

            train_transform.transforms.extend([
                transforms.Lambda(downsampling),
                transforms.CenterCrop(size=(400, 250)),
            ])
            test_transform.transforms.extend([
                transforms.Lambda(downsampling),
                transforms.CenterCrop(size=(400, 250))
            ])
        else:
            train_transform.transforms.extend([
                transforms.CenterCrop(size=(850, 550)),
            ])
            test_transform.transforms.extend([
                transforms.CenterCrop(size=(850, 550))
            ])
    else:
        train_transform.transforms.extend([
            transforms.Resize(size=256),
            transforms.RandomCrop(size=224)
        ])
        test_transform.transforms.extend([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224)
        ])

    train_transform.transforms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    for transform in [train_transform, test_transform]:
        transform.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return train_transform, test_transform
