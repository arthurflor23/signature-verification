## Decision Tree (CART, C4.5 and Random Forest)

Intelligence Computation and Pattern Recognition class projects, master degree, UPE - POLI (2018.2)

### Requirements

The code is written in Python 3. The extra libraries are required:

* Graphviz
* OpenCV
* Theano ¹
* Lasagne ²

¹²They can be installed by running the following commands:

``` 
pip install theano https://github.com/Lasagne/Lasagne/archive/master.zip
```

This project was tested on Arch Linux 64bits. This code can be used with or without GPUs - to use a GPU with Theano, follow the instructions in this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html). Note that Theano takes time to compile the model, so it is much faster to instantiate the model once and run forward propagation for many images (instead of calling many times a script that instantiates the model and run forward propagation for a single image).

### Usage

To run the project, execute `python main.py`.
The following code (from main.py) shows how to load, pre-process a signature, extract features and classifier.

### Datasets

* CNN model (to extract features): GPDS from [sigver_wiwd](https://github.com/luizgh/sigver_wiwd) repository
* Chars74K image dataset - University of Surrey [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
* ICDAR 2009 Signature Verification Competition from [SigComp2009](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
