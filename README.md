# MSDA
### A soft version of MSDA

This repo has implemented digits classification experiments in the paper: Multiple Source Domain Adaptation with Adversarial Learning[https://arxiv.org/abs/1705.09684]

![model](https://github.com/daoyuan98/MSDA/blob/master/images/model.png)

Thanks to the code from https://github.com/pumpikano/tf-dann , This repo has referenced much of the code there and this code is currently a simple extension from his code.

There a lot of codes that can be written more elegently so if you are interested, feel free to pull requests.

## Experiments Results.
On digits Classifaction, I have runed two experiments and have received satisfactory results. 

#### 1. Sv+Mm+Sy-->Mt
Domain Accurarcy and Iteration.
![1_dacc](https://github.com/daoyuan98/MSDA/blob/master/images/1_d_acc.png)

Model Accurarcy on Target Domain
![1_tar_acc](https://github.com/daoyuan98/MSDA/blob/master/images/1_tar_acc.png)

#### 2. Sv+Mt+Sy-->Mm
Domain Accuracy and Iteration
![2_dacc](https://github.com/daoyuan98/MSDA/blob/master/images/2_d_acc.png)

Model Accurarcy on Target Domain
![2_tar_acc](https://github.com/daoyuan98/MSDA/blob/master/images/2_tar_acc.png)
