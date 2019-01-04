# MSDA
### A soft version of MSDA

**Python 3.6 + Tensorflow 1.12.0** 

This repo has implemented digits classification experiments in the paper: Multiple Source Domain Adaptation with Adversarial Learning[https://arxiv.org/abs/1705.09684]
There a lot of codes that can be written more elegently so if you are interested, feel free to pull requests.

<center><img src="https://github.com/daoyuan98/MSDA/blob/master/images/model.png" width="800"></center>

## Acknowledgement
Thanks to the code from https://github.com/pumpikano/tf-dann , This repo has referenced much of the code there and this code is currently a simple extension from his code.

## Experiments Results
On digits Classifaction, I have carried out two experiments and have received satisfactory results. 

#### 1. Sv+Mm+Sy-->Mt
Domain Accurarcy and Iteration（left） and Model Accurarcy on Target Domain(right)
<div align="center">
    <img src="https://github.com/daoyuan98/MSDA/blob/master/images/1_d_acc.png" width="390"/><img src="https://github.com/daoyuan98/MSDA/blob/master/images/1_tar_acc.png" width="390"/>
</div>

#### 2. Sv+Mt+Sy-->Mm
Domain Accurarcy and Iteration（left） and Model Accurarcy on Target Domain(right)
<div align="center">
    <img src="https://github.com/daoyuan98/MSDA/blob/master/images/2_d_acc.png" width="390"/><img src="https://github.com/daoyuan98/MSDA/blob/master/images/2_tar_acc.png" width="390"/>
</div>

## Discussion
* I didn't put forward the 3rd experiment in the paper because I found that even after many many iterations(>130k), the model is overfitting but the accurarcy(~.764) is still far from that in the paper(.818). 
* The different training epochs may depend on the difficulty of different tasks.

## Any advice and comments are welcome!
