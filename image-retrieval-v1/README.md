# Image retrieval first approach
## Installation
Create a conda environment by running

```
conda create --name image-retrieval-v1 python=3.8
```
Then, activate the environment

```
conda activate image-retrieval-v1
```
install the dependencies

```
pip install -r image-retrieval-v1/requirements.txt 
```

setup `image-retrieval-v1` environment as Python interpreter in Visual Studio. Follow [this instructions](https://code.visualstudio.com/docs/python/environments#:~:text=To%20do%20so%2C%20open%20the,Settings%2C%20with%20the%20appropriate%20interpreter) 

and create a launch configuration in `launch.json` file pointing to the python file you want to execute. Example: 
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Retrieval",
            "type": "python",
            "request": "launch",
            "program": "image-retrieval-v1/main_jordi.py",
            "console": "integratedTerminal"
        }
    ]
}
```

## References
https://colab.research.google.com/drive/1fSAPnQrlEhIPFsb-2TrvI69qZ6qU5vKy?usp=sharing

