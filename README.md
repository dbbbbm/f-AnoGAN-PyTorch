# f-AnoGAN-PyTorch
An unofficial implementation of f-AnoGAN in PyTorch.

## Reference
- **Official TensorFlow implementation**:
https://github.com/tSchlegl/f-AnoGAN
- **Paper**: f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks
https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640
- **WGAN-GP-PyTorch**: https://github.com/jalola/improved-wgan-pytorch

## Dataset
This implementation performs anomaly detection on CIFAR-10. In the common setting we treat one class of CIFAR-10 as normal class and other 9 classes as anomalies. You can specify which class is considered as normal when running fanogan.py in command line by setting the `--class` argument.

## Usage
- Train a GAN

        python fanogan.py --stage 1 --class NORMAL_CLASS

- Train an encoder

        python fanogan.py --stage 2 --class NORMAL_CLASS

- Evaluation

        python fanogan.py --eval --class NORMAL_CLASS
