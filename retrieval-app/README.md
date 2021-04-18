# RetrievalApp

The Frontend application for the Image Retrieval system. The application is developed using the Angular 11 framework.

Examples of use:

* Retrieve jeans (image from Deep Fashion database)<br>
<img src="../docs/imgs/frontend-example-jeans.gif" width="600"/>
* Retrieve sweaters (image from Deep Fashion database)<br>
  <img src="../docs/imgs/frontend-example-sweater.gif" width="600"/>
* Retrieve T-shirt (image from Fashion Product database)<br>
  <img src="../docs/imgs/frontend-example-t-shirt-product_fashion.gif" width="600"/>

## Requirements for development environment

### Install npm

The easiest way to manage different node version in a development machine is making use of NVM (node version manager).

Use the information in their [GitHub](https://github.com/nvm-sh/nvm#installing-and-updating) to install NVM and install the node version v14.15.5.

### Install dependencies

In the folder `retrieval-app` execute:

```
npm install
```
This command will download all the dependencies defined in `package.json`.

### Development server

Run `ng serve --proxy-config proxy.conf.json` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

### Local Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory. Use the `--prod` flag for a production build.

## Docker build

Generate the Frontend docker image individually by executing the following command in the repository root:
```
docker build -f Dockerfile.app .
```

It is also possible to build the image with Docker Compose by executing the following command in the repository root:

```
docker-compose build app
```

### Proxy

The docker image is based on NGINX server so frontend can be served with a simple static file server, but also is included a proxy, that redirects requests from Frontend to the API docker container and avoid CORS problems. 

### Execute API standalone

Execute the Frontend with Docker Compose.

First of all export the environment variables DATASET_ROOT and WORKDIR_ROOT.

* DATASET_ROOT: environment variable that must point to the Fashion Product dataset root (see main README.md). It should be: ${PROJECT_ROOT}/datasets/Fashion_Product_Full/fashion-dataset
* WORKDIR_ROOT: environment variable that must point to the workdir. It should be: ${PROJECT_ROOT}/dataset/Fashion_Product_Full_Workir
* PROJECT_ROOT: environment variable pointing to the root of the repository.

So before, executing the docker-compose up, execute:

```
export PROJECT_ROOT= {POINT TO THE REPOSITORY ROOT}

export DATASET_ROOT=${PROJECT_ROOT}/datasets/Fashion_Product_Full/ && export WORKDIR_ROOT=${PROJECT_ROOT}/datasets/Fashion_Product_Full_Workdir

docker-compose up app

```
