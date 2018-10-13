# EyeFlow
[![Build Status](https://travis-ci.org/zperkowski/EyeFlow.svg?branch=master)](https://travis-ci.org/zperkowski/EyeFlow)

Finding choroids of the eye on a picture obtained by ophthalmoscopy.

## Used technologies
 * TensorFlow 1.7.0
 * Python 3.6.5
 * matplotlib 2.2.2
 * numpy 1.14.2

# Instalation

[TensorFlow](https://www.tensorflow.org/install/)

```
pip install matplotlib numpy scikit-image scipy
```

# Data sets
The main data set can be downloaded from [Friedrich-Alexander-Universität Erlangen-Nürnberg site](https://www5.cs.fau.de/research/data/fundus-images/).
The program with a default configuration uses pictures from: [Download the whole dataset (~73 Mb)](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip).
Extract the archive and move folders `images`, `manual1`, and `mask` to `data` direcory.

# How to run
If used `Virtualenv` from TensorFlow instalation guide:

```
cd EyeFlow/
source ./bin/activate
python3 eyeflow.py
```
