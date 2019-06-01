# EyeFlow
[![Build Status](https://travis-ci.org/zperkowski/EyeFlow.svg?branch=master)](https://travis-ci.org/zperkowski/EyeFlow)
[![codecov](https://codecov.io/gh/zperkowski/EyeFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/zperkowski/EyeFlow)

Finding choroids of the eye on a picture obtained by ophthalmoscopy.

## Used technologies
 * TensorFlow 1.7.0
 * Python 3.6.5
 * matplotlib 2.2.2
 * numpy 1.14.2

# Instalation

* [TensorFlow](https://www.tensorflow.org/install/)

```
cd EyeFlow/
source ./bin/activate
pip install -r requirements.txt
```

# Data sets
The dataset can be downloaded from [Friedrich-Alexander-Universität Erlangen-Nürnberg site](https://www5.cs.fau.de/research/data/fundus-images/).
EyeFlow with the default configuration uses pictures from: [Download the whole dataset (~73 Mb)](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip).
Extract the archive and move folders `images`, `manual1`, and `mask` to `data` direcory in root directory of the project.

# How to run

```
cd EyeFlow/
source ./bin/activate
python3 eyeflow.py
```
