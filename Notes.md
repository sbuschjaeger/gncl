To dos
    - For BNNs E2E seems to be better
    - For MobileNet GNCL seems to be better
    - EfficientNet V7 / MobileNet v3??

Some random notes

    - Ensembles seem to work better in the classic regime of loss > 0 and for modern Deep Learning models with loss = 0 they do not work well. The formulas clearly show this
    - Do we still train modern DL architectures (e.g. mobilenet) on CIFAR100/ImageNet do achieve 0 loss or is this impossible if we perform data augmentation etc?