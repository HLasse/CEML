# CEML
Resources for exan in the course on Constructive and Explainable Machine Learning.

The Task was to train a CNN on CIFAR-10 and explore which insights could be gathered from gradient-based pixel attributions models (using [Captum](https://captum.ai)) and nearest neighbors. 

## Setup 
Create virtual environment

Install requirements ```pip install -r requirements.txt```


## Model training and explaining
Train model by running ```python cifar_train.py```. Alternatively, skip this step and use the pretrained model (*model.h5*) instead. 

To generate explanations and extract nearest neighbors, run ```python gradient_exp.py```. This will create an ```img``` folder containing subfolders for output from Integrated Gradients, Guided Grad-CAM, and Saliency Maps. 




