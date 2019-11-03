DeepFake Detection
===

DeepFakes have been in the news for over a year now, mostly for all the wrong reasons. This repository is home to Pytorch implementations of various deepfake detection algorithms, tested on Google AI's [Deep Fake Detection Dataset](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html).


## Algorithms


1. ### [MesoNet](https://github.com/DariusAf/MesoNet)
    
    *Data* : Due to computational constraints, trained the MesoInception4 network on a subset of the DeepFake Detection dataset(363 videos from both real and manipulated classes).

    *Results* : The data was trained for 30 epochs using all default hyperparamametes and network as mentioned in the [original paper](https://arxiv.org/abs/1809.00888).
    
        Training Accuracy: 93.75% 
        Test Accuracy: 90.42%
    

