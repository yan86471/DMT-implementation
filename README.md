# DMT
## Introduction

A implementation of [Disentangled Makeup Transfer with Generative Adversarial Network](https://arxiv.org/abs/1907.01144).

---

## Training
1. Download  MT (Makeup Transfer) dataset from [here](http://liusi-group.com/projects/BeautyGAN).

2. Put MT (Makeup Transfer) dataset to `.\data\RawData`.
    Your data path will like this:
```
.\data\RawData\images\makeup\*.png
.\data\RawData\images\non-makeup\*.png

.\data\RawData\segs\makeup\*.png
.\data\RawData\segs\non-makeup\*.png
```

3. run `python convert.py`

4. Modify train.py and start training.  
    `python train.py` 

5. run `python export.py` and you will get h5 model in `.\Export`.

## Demo
1. make sure you have run `python export.py` to get h5 model.

2. Modify demo.py and run `python demo.py`, you will find the transfer result in `.\Transfer`.  


## Some issues to know
1. The test environment is
    - Python 3.7
    - tensorflow-gpu 2.0.0
    - tensorflow-addons 0.7.1
    - imgaug 0.4.0

2. This is still not a completed implementation, but almost 95% is the same as paper described.