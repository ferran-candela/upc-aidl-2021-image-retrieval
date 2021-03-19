# Setting for VSCode

```
    {
        "name": "Train",
        "type": "python",
        "request": "launch",
        "program": "/home/fcandela/src/upc/upc-jmc-project/imageretrieval/src/train.py",
        "console": "integratedTerminal",
        "env": {
            "DEBUG": "True",
            "DATASET_BASE_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset",
            "DATASET_LABELS_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset/styles.csv",
            "WORK_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset",
            "LOG_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset/log/",
            "TRAIN_SIZE": "all",
            "TEST_VALIDATE_SIZE": ""
        }
    }

    {
            "name": "Extract features",
            "type": "python",
            "request": "launch",
            "program": "/home/fcandela/src/upc/upc-jmc-project/imageretrieval/src/features.py",
            "console": "integratedTerminal",
            "env": {
                "DEBUG": "True",
                "DATASET_BASE_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset",
                "DATASET_LABELS_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset/styles.csv",
                "WORK_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset",
                "LOG_DIR": "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset/log/",
                "TRAIN_SIZE": "500",
                "TEST_VALIDATE_SIZE": "500"
            }
        }
```

# Command line execution for training

Activate the conda environment, setup environment vars and execute train.py.

```
source /home/fcandela/opt/miniconda3/bin/activate &&
conda activate image-retrieval-v1

export DEBUG=True &&
export DATASET_BASE_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset &&
export DATASET_LABELS_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset/styles.csv &&
export WORK_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset &&
export LOG_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset/log/ &&
export TRAIN_SIZE=all &&
export TEST_VALIDATE_SIZE=0 &&
python /home/fcandela/src/upc/upc-jmc-project/imageretrieval/src/train.py
```

# Command line execution for feature extraction

Activate the conda environment, setup environment vars and execute features.py.

```
source /home/fcandela/opt/miniconda3/bin/activate &&
conda activate image-retrieval-v1

export DEBUG=True &&
export DATASET_BASE_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset &&
export DATASET_LABELS_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset/styles.csv &&
export WORK_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset &&
export LOG_DIR=/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset/log/ &&
export TRAIN_SIZE=all &&
export TEST_VALIDATE_SIZE=0 &&
python /home/fcandela/src/upc/upc-jmc-project/imageretrieval/src/features.py
```