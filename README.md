# Handwritten Text Recognition Codes

This repository contains codes to Handwritten Text Recognition to be used with CPUs.
The code, in general, is open to new implementations of techniques, among these are classifiers using Decision Tree (C4.5, CART, Random Forest) and Neural Networks with Connectionist Temporal Classification.

Reference repositories:
Neural Network for Signature Characteristic Extraction: [luizgh](https://github.com/luizgh/sigver_wiwd)
Handwritten Text Recognition with Word Beam Search: [githubharald](https://github.com/githubharald/SimpleHTR)

## Datasets

* CNN model (to extract features): GPDS from [sigver_wiwd](https://github.com/luizgh/sigver_wiwd) repository
* Signatures to classifier: ICDAR 2009 Signature Verification Competition from [SigComp2009](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
* Characteres to classifier (tests during the process): [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)

## Installation

This project was tested on Arch Linux 64bits and was developed to be out of box (changes the main.py file)

### Pre-requisites

The code is written in Python 3. It's recommended to use the Anaconda python distribution ([link](https://www.anaconda.com/download/)).

The extra libraries are required:

* Graphviz
* OpenCV
* Theano
* Lasagne
* TensorFlow

They can be installed by running the following commands:

```
conda install opencv theano
pip install graphviz https://github.com/Lasagne/Lasagne/archive/master.zip
```