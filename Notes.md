# TO DOs

- Do we still train modern DL architectures (e.g. mobilenet) on CIFAR100/ImageNet do achieve 0 loss or is this impossible if we perform data augmentation etc?
  - We can check this, since we MobileNetv3 now!

## Some random notes

### CIFAR100
  - ResNet trains much faster than VGG, but achieves the same accuracy
  - ResNet with ~88K Parameters has 55% accuracy. Ensemble of 16 models gets us to ~ 67
- EfficientNet (https://github.com/lukemelas/EfficientNet-PyTorch) is the newest shit, but does not work well on smaller image sizes. It is optimized for imagenet with `img_size = 224`. This sucks for CIFAR / Fashion
- MobileNet is also nice (https://github.com/rwightman/gen-efficientnet-pytorch), but has a similar problem compared to EfficientNet. 
  - I found I rather simple but general MobileNetV3 implementation (https://github.com/ShowLo/MobileNetV3) which also supports CIFAR / Fashion using a different stride.
  - MobileNet needs SGD!!
  - M = 16, small MobileNetV3 ==> 3 min / epoch ==> 10h / model + 12 GB :(
  - Gefällt mir nicht so richtig tbh
- A simple VGGNet with roughly 1.5 Mio params is MUCH FASTER than MobileNetV3 (8 secs / epoch vs 20 secs / epoch) and gets a similar accuracy on Fashion?!
  - vgg_model, hidden_size = 1024, model_type = "float", depth = 2, n_channels = 16
- Ensembles seem to work better in the classic regime of loss > 0 and for modern Deep Learning models with loss = 0 they do not work well. The formulas clearly show this
- There is no diffrence between upper and exact minimization of GNCL. 
- BNN braucht ADAM?