# Flask API for image retrieval

The Rest API for image retrieval has been implemented using Flask and Flask-RESTful. It is composed by 3 endpoints:

* Models endpoint
* Search endpoint
* Image endpoint

## Models endpoint

This endpoint returns the available models that can be used to perform a retrieval.

* **URL**

    /api/models

* **Headers:**

    Accept=application/json

* **Method:**

    `GET`
  
*  **URL Params**

    None

* **Data Params**

    None

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** 
    ```json
        {
            "models": [
                "resnet50_custom",
                "vgg16",
                "resnet50",
                "inception_v3",
                "inception_resnet_v2",
                "densenet161",
                "efficient_net_b4"
            ]
        }
    ```
 
* **Error Response:**

  None

## Search endpoint

This endpoint returns the `topK` ranking of database image ids for the selected `model` and the provided `image`.

* **URL**

    /api/search

* **Method:**

    `POST`

* **Headers:**
  
  Content-Type=multipart/form-data
  Accept=application/json

*  **URL Params**

    None

* **Data Params**

    Form params:

    `image=[Blob]` > The image of interest. It must be a jpg or jpeg.<br/>
    `model=[String]` > The model to be used in retrieval.<br/>
    `topK=[Integer]` > The number of results to obtain.

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** 
    ```json
    {
        "success": true,
        "ranking": [
            7747,
            12203,
            12204,
            3780,
            2764
        ]
    }
    ```
 
* **Error Response:**

    * **Code:** 400 - BAD REQUEST <br />
    **Content:** 
        ```json
        {
            "success": false
        }
        ```

## Images endpoint

This endpoint returns an image given the correspoding `id`.

* **URL**

    /api/images/${id}

* **Headers:**

    Accept=image/jpeg

* **Method:**

    `GET`
  
*  **URL Params**

    id -> The image id to retrieve.

* **Data Params**

    None

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** 
    The image file.
 
* **Error Response:**

    * **Code:** 404 - NOT FOUND <br />
    **Content:** 
        None

# Postman project

In the folder `docs` it is included a Postman collection file with an example for each endpoint. Just use the `Import` feature in Postman application, run the API in the `localhost:5000` and try it by yourself.

# Project structure

```
    .
    ├── docs                            # Documentation  
    │   └── postman                     # Postman project to import  
    ├── src                             # Source files  
    │   ├── app                         
    │   │   ├── api                     
    │   │   │   ├── controller.py       # API endpoints  
    │   │   │   └── schemas.py          # Model schemas  
    │   │   ├── common                  
    │   │   │   └── error_handling.py   # Error handling  
    │   │   ├── __init__.py             # Flask app creation   
    │   │   └── ext.py                  # Marshmallow config
    │   └── config                      
    │       └── default.py              # Config variables
    └── README.md                       # API Readme
```

# Execution in VS Code

```
    {
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Image Retrieval API Rest",
            "type": "python",
            "request": "launch",
            "module": "flask",
            "env": {
                "FLASK_APP": "entrypoint.py",
                "FLASK_ENV": "development",
                "FLASK_DEBUG": "0",
                "DEVICE": "cpu",
                "DEBUG": "True",
                "DATASET_BASE_DIR": "${PROJECT_ROOT}/datasets/Fashion_Product_Full/fashion-dataset",
                "DATASET_LABELS_DIR": "${PROJECT_ROOT}/datasets/Fashion_Product_Full/fashion-dataset/styles.csv",
                "WORK_DIR": "${PROJECT_ROOT}/datasets/Fashion_Product_Full_Workdir",
                "LOG_DIR": "${PROJECT_ROOT}/datasets/Fashion_Product_Full_Workdir/log/"
            },
            "args": [
                "run",
                "--no-debugger",
                "--no-reload"
            ],
            "jinja": true
        }]
    }
```
# Execution in Docker

## Preconditions

### Step 1: Get Docker

The first step is installing the Docker Engine. Follow the steps in the
official page: 

[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)  
  
  
### Step 2: Get Docker Compose
Then install Docker Compose. 

[https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)


## Build API 

Generate the API docker image individually by executing the following command in the repository root:
```
docker build -f Dockerfile.api .
```

It is also possible to build the image with Docker Compose by executing the following command in the repository root:

```
docker-compose build api
```

## Execute API standalone

Execute the API with Docker Compose.

First of all export the environment variables DATASET_ROOT and WORKDIR_ROOT.

* DATASET_ROOT: environment variable that must point to the Fashion Product dataset root (see main README.md). It should be: ${PROJECT_ROOT}/datasets/Fashion_Product_Full/fashion-dataset
* WORKDIR_ROOT: environment variable that must point to the workdir. It should be: ${PROJECT_ROOT}/dataset/Fashion_Product_Full_Workir
* PROJECT_ROOT: environment variable pointing to the root of the repository.

So before, executing the docker-compose up, execute:

```
export PROJECT_ROOT= {POINT TO THE REPOSITORY ROOT}

export DATASET_ROOT=${PROJECT_ROOT}/datasets/Fashion_Product_Full/ && export WORKDIR_ROOT=${PROJECT_ROOT}/datasets/Fashion_Product_Full_Workdir

docker-compose up api

```

# Retrieval Engine

The API makes use of the Image Retrieval engine, the core of the  retrieval system. The Engine is connected with the ModelManager and the FeatureManager to start up the different models and features stored during the training and database preprocessing. So the output of the feature extraction step is the Image Database that will be used to retrieve similar results to user requests.

The API controller exposes the `query` method to perform a retrieval that basically perform the following steps:

1. Select the model to be used from memory.
2. Preprocess the query image with the sizes configured for the selected model.
    1. Resizes the image to fit the model input size.
    2. Center crop with the model input size.
    3. Normalize.
3. Compute the image features using the selected feature extractor model.
4. Compute the cosine similarity between the query image and all the precomputed features of the Database.
5. Rank and select the first topK.
6. Return an array of database image ids.

