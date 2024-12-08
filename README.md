# Deeplearning Homework

## Team name

**mélytanulóbúvárok**

## Team members

- **Botos András** (`VI0W9O`)
- **Naán Gergely** (`QMSC7T`)

## Project Description

**Vision transformers for medical image segmentation**

```
Transformer networks revolutionized deep learning, especially in the field of natural language processing, and are gaining much attention in computer vision. They offer great advantages over convolutional networks (CNN), such as higher flexibility, less sensitivity to hyperparameters, and the ability to effectively capture both local and global features in the input data. However, in some cases CNNs still excel, and for some tasks such as segmentation even transformer-based architectures use convolutional layers.

Task of the students: explore transformer networks for medical image segmentation. Investigate open-source implementations, find and test a CNN-based baseline solution, and train 1-2 networks with different architectures on a cardiac MRI segmentation dataset (other, even non medica datasets also allowed). Pure-transformer or hybrid transformer-cnn or fully convolutional architectures are ok as well, one must be transformer based. Compare the selected networks by accuracy, throughput, sensitivity to hyperparameter changes, ease of implementation and training, etc.
```

**We choose an other, non medical dataset (as accordance with the description).**

Dataset chosen: [Pascal 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

# Milestonse 1

## Files

Initialization scripts:

*These scripts are run automatically when you start the container.*

- `data_preparation/download_data.py`: Downloads the dataset and extracts it into the `data` folder.
- `data_preparation/convert_dataset.py`: Converts the xml files from the dataset to a single jsonlines file, which is necessary for the `data_analysis.ipynb` notebook.
- `data_preparation/object_picture_relation_finder.py`: Creates a new `data/relations` folder required by `even_distribution_creator.ipynb`.

Misc files:

- `data_preparation/requirements.txt`: Contains the required Python packages.
- `data_preparation/Dockerfile`: Describes the docker image.
- `data_preparation/run.sh`: Runs the initialization scripts in sequence, and then starts Jupyter Lab.

Notebooks:

- `data_preparation/data_analysis.ipynb`: This notebook guides us through the exploratory data analysis.
- `data_preparation/size_converter.ipynb`: Fixes format of xml files in data for the following notebook.
- `data_preparation/even_distribution_creator.ipynb`: Our doomed beautiful endeavor at lowering class imbalance, demonstrating modest\* improvements.

\*: standard deviation of classes lowered from 340 to 287

## Related works

We were inspired by the following works:

- https://github.com/farakiko/ImageSegmentationPASCAL

- https://www.ibm.com/topics/image-segmentation

- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf

- https://www.superannotate.com/blog/image-segmentation-for-machine-learning

We used the following pages to complete our code:

- https://www.geeksforgeeks.org/read-json-file-using-python/

- https://github.com/albumentations-team/albumentations

- https://albumentations.ai/docs/examples/example/

## How to run

1. Start the container:

```bash
docker compose up
```

2. Open [Jupyter Lab](http://127.0.0.1:8888/lab)

3. Enjoy the prepared notebooks

4. Profit

### Requirements

- A recent version of docker and docker compose.
- ~6 GB of free space: The docker image takes about 1 GB of storage, while the compressed+extracted dataset takes about 4.5 GB.
- Port 8888 open.


# Milestonse 2

Initialization scripts:

*These scripts are run automatically when you start the container.*

- `training/copy_image_mask.py`: Organizes the images and masks, preparing them and placing them in the designated folder..
- `training/data_split.py`: Splits the data into the well-known train, test, validation sets.
- `training/dataset.py`: Creates loaders for the 2 CNN models.
- `training/train_cnn.py`: Trains a Fully-Convolutional Network model with a ResNet-50 backbone.
- `training/test_cnn.py`: Tests a Fully-Convolutional Network model with a ResNet-50 backbone.
- `training/train_cnn_v2.py`: Trains a Fully-Convolutional Network model with a ResNet-101 backbone.
- `training/test_cnn_v2.py`: Tests a Fully-Convolutional Network model with a ResNet-101 backbone.
- `training/train_vit.py`: Trains a Vision Transformer model which is based on the MiT-B0 architecture.
- `training/test_vit.py`: Tests a Vision Transformer model which is based on the MiT-B0 architecture.


## Related works

We were inspired by the following works:

- https://huggingface.co/docs/transformers/en/tasks/semantic_segmentation

## How to run

1. Navigate into the training folder

2. Start the container:

```bash
docker compose up
```

- With this pipeline, training and testing are performed for all three models.

### Requirements

- Milestone 1 must be run beforehand.
- A recent version of docker and docker compose.
- ~22 GB of RAM is needed: The docker image takes about 11 GB of storage


# Milestonse 3 / Final submission

## Related works

## How to run frontend 

1. Navigate into the training folder

2. Start the container:

```bash
docker compose up
```

- With this container you can start a gradio inference frontend to try out our two cnn and one vit models.

### Requirements for frontend

- 8 GB of storage
- 2.5 GB of free ram recommended
- Milestone 1 and 2 should be run beforehand. Alternatively you may download our models from our GitHub releases page and put them in the data folder (extracting vit.zip).
- A recent version of docker and docker compose.
