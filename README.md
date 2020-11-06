# Auxiliary Losses
I always find myself wondering, what would happen if we introduce certain additional loss terms in the objective. Namely, assume - <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCE%7D" />  is the usual cross entropy loss in classification tasks. Now, assume that we have great idea, and would like to introduce it as an additional loss term - <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCustom%7D" />  
i.e. to have the following loss: <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BTotal%7D%20%3D%20%5Cmathcal%7BL%7D_%7BCE%7D%20&plus;%20%5Cmathcal%7BL%7D_%7BCustom%7D" /> <br>
What would be the effect of <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCustom%7D" /> ?
This repo will serve as a codebase for this type of experiments. <br>
<br>

**Architectures**
- [x] Basic MLP & Convnets
- [x] Resnets
- [ ] LSTMs
- [ ] Transformers

<br>

**Tasks**
- [x] Image Classification (ImageNet, CIFAR10, CIFAR100, MNIST)
- [x] Time Series Classification (UCR) 
- [ ] Time Series Regression
- [ ] Object Detection
<br> 

**Losses**: 
- [x] Augmentation + KL-divergence [1]
- [ ] Contrastive loss 
- [x] Adversarial Examples

<br> 

**Benchmarks**: 
- [x] Gaussian noise
- [ ] Common corruptions
- [ ] Adversarial examples

<br> 


Example training runs: <br>
`python training.py --model=convnet --dataset=mnist --custom-loss=gaussian-kl_0.1 --device=cpu --tqdm` <br>
`python training.py --model=mlp --dataset=UCR_ECG200 --custom-loss=gaussian-kl_0.1 --tqdm` <br>
`python training.py --model=convnet --dataset=mnist --custom-loss=adversarial-attack --tqdm --loss-config-file=./configs/pgd_2_0.5.json` <br>
`python training.py --model=convnet --dataset=mnist --custom-loss=adversarial-attack --device=cpu --tqdm --loss-config-file=./configs/pgd_2_0.5.json` <br>

Evaluation: <br>
`python evaluation.py --model=mlp --dataset=UCR_ECG200 --eval-benchmark=gaussian_0.1 --tqdm --device=cpu --eval-run=./results/UCR_ECG200_mlp_Adam_0.001_gaussian-kl_0.1`

Will add further loss terms and some example runs soon. <br>


1- Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. Unsupervised data augmentation. arXiv preprint arXiv:1904.12848, 2019.
