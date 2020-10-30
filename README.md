# Loss Test Codebase
I always find myself wondering, what would happen if we introduce certain additional loss terms in the objective. Namely, assume - <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCE%7D" />  is the usual cross entropy loss in classification tasks. Now, assume that we have great idea, and would like to introduce it as an additional loss term - <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCustom%7D" />  
i.e., 
<img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BTotal%7D%20%3D%20%5Cmathcal%7BL%7D_%7BCE%7D%20&plus;%20%5Cmathcal%7BL%7D_%7BCustom%7D" /> 
What would be the effect of? <img src="https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BCustom%7D" />  <br>
This repo will serve as a codebase for this type of experiments. <br>
Currently, <br>
**Architectures** : MLP, ResNets<br>
**Tasks**: Image Classification (ImageNet, CIFAR10, MNIST) <br> 

Example run: <br>
`python run.py --model=resnet18 --dataset=cifar10 --add-loss=aug-kl-div`

Will add further loss terms and some example runs soon. <br>
