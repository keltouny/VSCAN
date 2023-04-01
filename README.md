# Uncertainty-Aware Structural Damage Warning System Using Deep Variational Composite Neural Networks
This is the code for the paper

Kareem Eltouny and Xiao Liang. Uncertainty-Aware Structural Damage Warning System Using Deep Variational Composite Neural Networks. Earthquake Engineering & Structural Dynamics (Under review).

![overview](https://github.com/keltouny/vscan/blob/main/figures/overview.jpg)

## Dependencies
- python 3.8
- CUDA 11.2
- tensorflow 2.6.0

## Get the data
We perform structural condition assessment on three real structures that were instrumented as part of California Strong Motion Instrumentation Program (CSMIP) with recordings during the 1994 Northridge earthquake (prior recordings are used for trianing and validation).
The selected buildings are the Van Nuys hotel (ID 24376), the Sherman Oaks commercial building (ID 24322), and the San Bernardino hospital (ID 23634).
The recordings can be obtained from [Center for Engineering Strong Motion Data](https://www.strongmotioncenter.org/).

## Contents description
Scripts for each case are located in independent directories. The code is identicial except for the "main.py" files.
Records from CSMIP should be placed in the "building/{stationID}/" directory.

Saved model weights are located in the "Weights" directory.

## Instructions

In each case directory, the main file should be ran for training and evaluating the network.

Edit the following lines to choose whether to train, test, or perform both:

```
train_model = False
test_model = True
```

Saved model weights are located in the "Weights" directory. To be used, they need to be placed in the corresponding case directory (with train=False).


## Citation
If you wish to cite the work, you may use the following:
```ruby
@article{author = {Eltouny, Kareem and Liang, Xiao},
    title = "{Uncertainty-Aware Structural Damage Warning System Using Deep Variational Composite Neural Networks}",
    journal = {Earthquake Engineering & Structural Dynamics},
    year = {under review},
}
```
