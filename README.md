# Deep Learning Image retrieval with Fashion Product database

The aim of the project is the creation of a tool that allows the full implementation and evaluation of an image retrieval web application using the [Fashion Product](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset) dataset as the database. DuSeveral existing pre-trained models are evaluated and also custom models are created from applying Transfer Learning from them.

The main idea is that a user can obtain the topK ranking products more similar to an image using one of the different models available.

The [Deep Fashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) database is used to evaluate the system with more heterogeneous images (real world images) and finally used to obtain the best model as it is depicted in the evaluation results.

The original code is the Final Project delivery for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 edition, authored by:

* Melania Sánchez
* Jordi Montaner
* Ferran Candela

Advised by professor Kevin McGuinness 


## Project overview

   <img src="docs/imgs/similarity_engine_diagram.png" width="800">

Documentation of the different parts of the project:
* [Data preparation](/imageretrieval/README.md/#datapreparation)
* [Training](/imageretrieval/README.md/#training)
* [Model](/imageretrieval/README.md/#model)
* [Feature extraction](/imageretrieval/README.md/#featureextraction)
* [Finetune](/imageretrieval/README.md/#finetune)
* [Evaluation](/imageretrieval/README.md/#evaluation)
* [Feature visualization](/imageretrieval/README.md/#featurevisualization)
* [API](/retrievalapi/README.md)
* [Frontend](/retrieval-app/README.md)

Documentation of the [experiments](#experiments):
1. [Pretrained model](#pretrainedmodel)
2. [Custom model Densenet161](#densenet161custom)
3. [Custom model Densenet161 with PCA finetune](#densenet161custompca)
4. [Evaluation with deep fashion](#evaluationdeepfashion)
5. [Custom model with deep fashion](#custommodeldeepfashion)
6. [Custom model with batchnorm](#custommodelbatchnorm)


## General configuration


## Fashion product dataset

It is necessary to remove the following lines in the 'styles.csv' since they are missing images.

```
39425,Men,Apparel,Topwear,Tshirts,Red,Spring,2013,Casual,U.S. Polo Assn. Men Red Polo T-Shirt
39401,Men,Apparel,Bottomwear,Jeans,Blue,Winter,2016,Casual,U.S. Polo Assn. Denim Co. Men Blue Slim Straight Fit Jeans
12347,Men,Apparel,Topwear,Suits,Red,Winter,2010,Casual,Fastrack Men Red Manhattan Regular Fit Solid Formal Shirt
39403,Men,Apparel,Topwear,Shirts,Black,Summer,2014.0,Casual,U.S. Polo Assn. Men Black Tailored Fit Casual Shirt
39410,Men,Apparel,Topwear,Shirts,Cream,Summer,2014,Casual,U.S. Polo Assn. Men Cream-Coloured Tailored Fit Casual Shirt
```

# Install dependencies

Step 1: Get Docker
```
The first step is installing the Docker Engine. Follow the steps in the
official page: 

[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
```

Step 2: Get Docker Compose
```
Then install Docker Compose. 

[https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)
```

# Build project

First of all export the environment variables DATASET_ROOT and WORKDIR_ROOT.


In the root folder execute:

```
docker-compose -p imageretrieval build
```

This will creaate the Docker image for the API and the Frontend (that also acts as a proxy for the API in the port 80).

# Experiments

## <a name="pretrainedmodel"></a>First experiment - Pretrained model Resnet50
This experiment was the first step of our retrieval engine, we build a system that used pretrained models.
The expectation was to first of all get to know the dataset `product fashion` and train it with pretrained model to be able to identify the best model. Also we took this opportunity to improve the transform operations that we need to apply to the input images to improve the accuracy.
We have performed this experiment over 6 different models:
* vgg16
* resnet50
* inception_v3
* inception_resnet_v2
* densenet161
* efficient_net_b4


### Hypothesis
The expectation from what we read is that the model that would behaves the best is resnet50

### Experiment setup
   <img src="docs/imgs/resnet50_architecture.webp" width="100"/>  
   Diagram from resnet50 architecture

### Results
### Conclusions

# <a name="experiments">Experiments

## <a name="densenet161custom">Second experiment - Customer model Densenet161

### Hypothesis
### Experiment setup
### Results
### Conclusions


## <a name="densenet161custompca">Third experiment - Customer model Densenet161 with finetune PCA

### Hypothesis
### Experiment setup
### Results
### Conclusions


## <a name="evaluationdeepfashion">Fourth experiment - Evaluation for deep fashion

### Hypothesis
### Experiment setup
### Results
### Conclusions


## <a name="custommodeldeepfashion">Fifth experiment - Evaluation for deep fashion

### Hypothesis
### Experiment setup
### Results
### Conclusions


## <a name="custommodelbatchnorm">Sixth experiment - Evaluation for deep fashion

### Hypothesis
### Experiment setup
### Results
### Conclusions

# Bibliography
* Resnet50 diagram from [cv-tricks](https://cv-tricks.com/keras/understand-implement-resnets/)