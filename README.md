# MSDA
### A soft version of MSDA

This repo has implemented digits classification experiments in the paper: Multiple Source Domain Adaptation with Adversarial Learning[https://arxiv.org/abs/1705.09684]

<center><img src="https://github.com/daoyuan98/MSDA/blob/master/images/model.png" width="600"></center>

Thanks to the code from https://github.com/pumpikano/tf-dann , This repo has referenced much of the code there and this code is currently a simple extension from his code.

There a lot of codes that can be written more elegently so if you are interested, feel free to pull requests.

## Experiments Results.
On digits Classifaction, I have runed two experiments and have received satisfactory results. 

#### 1. Sv+Mm+Sy-->Mt
Domain Accurarcy and Iteration（left） and Model Accurarcy on Target Domain(right)
<div align="center">
    <img src="https://github.com/daoyuan98/MSDA/blob/master/images/1_d_acc.png" width="200"/><img src="https://github.com/daoyuan98/MSDA/blob/master/images/1_tar_acc.png" width="200"/>
</div>

#### 2. Sv+Mt+Sy-->Mm
Domain Accurarcy and Iteration（left） and Model Accurarcy on Target Domain(right)
<div align="center">
    <img src="https://github.com/daoyuan98/MSDA/blob/master/images/2_d_acc.png" width="310"/><img src="https://github.com/daoyuan98/MSDA/blob/master/images/2_tar_acc.png" width="310"/>
</div>
