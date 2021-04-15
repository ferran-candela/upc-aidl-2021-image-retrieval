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

## Postman project

In the folder `docs` it is included a Postman collection file with an example for each endpoint. Just use the `Import` feature in Postman application, run the API in the `localhost:5000` and try it by yourself.