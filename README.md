# deeplearning-homework

Team name: **mélytanulóbúvárok**

Team members:
- **Botos András** (`VI0W9O`)
- **Naán Gergely** (`QMSC7T`)

Project Description:

**Vision transformers for medical image segmentation**

```
Transformer networks revolutionized deep learning, especially in the field of natural language processing, and are gaining much attention in computer vision. They offer great advantages over convolutional networks (CNN), such as higher flexibility, less sensitivity to hyperparameters, and the ability to effectively capture both local and global features in the input data. However, in some cases CNNs still excel, and for some tasks such as segmentation even transformer-based architectures use convolutional layers.

Task of the students: explore transformer networks for medical image segmentation. Investigate open-source implementations, find and test a CNN-based baseline solution, and train 1-2 networks with different architectures on a cardiac MRI segmentation dataset (other, even non medica datasets also allowed). Pure-transformer or hybrid transformer-cnn or fully convolutional architectures are ok as well, one must be transformer based. Compare the selected networks by accuracy, throughput, sensitivity to hyperparameter changes, ease of implementation and training, etc.
```

**We choose an other, non medical dataset (as accordance with the description).**

Dataset chosen: [Pascal 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

Files:

- `data_preparation/convert_dataset.py`: Converts the xml files from the dataset to a single jsonlines file. Needed for the `data_analysis.ipynb` notebook.
- `data_preparation/download_data.py`: Downloads the dataset and extracts it to the `data` folder.
- `data_preparation/requirements.txt`: Contains the required python packages.
- `data_preparation/data_analysis.ipynb`: This notebook contains the exploratory data analysis.
- TODO: ...

Related works:

*Note: It was nowhere mentioned in the lesson materials that we would have to explore related works by this deadline. To the best of our knowledge this was also also not mentioned during lectures.*

**TODO: related works goes here**

How to run:

1. Start container:

```bash
docker compose up
```

2. Open [Jupyter Lab](http://127.0.0.1:8888/lab)

3. ???

4. Profit

**TODO**