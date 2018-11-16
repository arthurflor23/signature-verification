# Signatures Classifier

This repository contains the code and instructions to use a trained CNN model with a random forest to signatures classifier.
The project was based on the model trained by [luizgh](https://github.com/luizgh/sigver_wiwd) with a Random Forest as the classifier at the end.

## Installation

### Pre-requisites

The code is written in Python 3. It's recommended to use the Anaconda python distribution ([link](https://www.anaconda.com/download/)).

The extra libraries are required:

* Graphviz
* OpenCV
* Theano
* Lasagne

They can be installed by running the following commands:

```
conda install opencv theano
pip install graphviz https://github.com/Lasagne/Lasagne/archive/master.zip
```

This project was tested on Arch Linux 64bits. This code can be used with or without GPUs - to use a GPU with Theano, follow the instructions in this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html). Note that Theano takes time to compile the model, so it is much faster to instantiate the model once and run forward propagation for many images (instead of calling many times a script that instantiates the model and run forward propagation for a single image).

### Downloading the project

All the material needed for the project is already in the repository. So, just clone (or download) this repository.

## Usage

The following code (from main.py) shows how to load, pre-process a signature, extract features and classifier.

## Datasets

* CNN model (to extract features): GPDS from [sigver_wiwd](https://github.com/luizgh/sigver_wiwd) repository
* Signatures to classifier: ICDAR 2009 Signature Verification Competition from [SigComp2009](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
* Characteres to classifier (tests during the process): [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)

## License

The source code is released under the BSD 2-clause license. Note that the trained model (CNN) used the GPDS dataset for training (which is restricted for non-comercial use).