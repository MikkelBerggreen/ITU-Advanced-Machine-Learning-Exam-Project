## Repository description

In this repository you will find our source code and all trained models. 

The source code can be found in the `src` folder, which contains four notebook and a `utils` folder. The latter contains the actual implementation of our model and various configurations and helper methods, which we import in our notebooks. The notebooks are:

1. `visualization.ipynb`: Contains the logic for preliminary visualizations of the dataset.
2. `main.ipynb`: Contains the logic for preprocessing the training data and training a model. To run different models change the configuration in `src/utils/config.py`.
3. `results.ipynb`: Contains the logic for preprocessing the test data and testing the model.
4. `Gabor.ipynb`: Contains the logic for computing results with the Gabor Wavelet Pyramid.

## Problem description - Modelling how the brain represents visual information

![](mp_brainvision.png)

Data science and machine learning approaches have huge potential to boost fundamental research of understanding the brain. 
In this project, we model the visual processing hierarchy of the brain by building a deep CNN network with a bio-inspired architecture to represent intermediate feature extracting with fMRI data of brain imaging studies. In particular, we want to look at the human fMRI response to visual stimulation and investigate which brain regions encode which type of visual information by predicting fMRI response from the model. As input for the study and also our model, we consider 
- the [Kay & Gallant image + fMRI dataset](https://crcns.org/data-sets/vc/vim-1/about-vim-1) (via mp-brainvision-load-kay-images.ipynb) or 
- [The Algonauts video + fMRI dataset](https://docs.google.com/forms/d/e/1FAIpQLScqEf-F5rAa82mc1_qbnoMonHVmfFk52kaCJQpTAkDU0V5vUg/viewform) (via mp-brainvision-load-algonauts-videos.ipynb).

This task is based on [the Algonauts challenge](http://algonauts.csail.mit.edu/challenge.html) of modelling how the human brain makes sense of the world. Thus here, you can learn more basics on brain imaging and imaging data processing.

### Main goals:

- Do a comprehensive pre-processing and some visualisation of the fMRI data. Can you see patterns in the data for either input category?
- Train a voxel-wise encoding model (deep CNN network) to predict fMRI responses in different brain regions from the stimulus data as input. You might consider vanilla versus pre-trained image processing networks. Can you identify an architecture (and meta-parameter settings) that predicts well and remains bio-inspired regarding the hierarchy?
- Compare predictions of the individual layers (model) with activations of different regions (imaging data), e.g. through heatmaps. How does the cortex hierarchy compare to the model's hierarchy? Do you observe any patterns?

### Optional:

- Can you suggest which brain regions preserve spatial information? Apply image transformations randomly before feeding to the model and observe the change in encoding accuracy.
- Can you identify in which regions the brain might encode categorical information? For this, you might compare representation similarities (e.g. via RSA).
