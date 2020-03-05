# LearnedSketch

This repository contains the code for the paper [Learning-Based Frequency Estimation Algorithms](https://openreview.net/pdf?id=r1lohoCqY7) in ICLR 2019.

If you find the code useful, please cite our paper!
```
@inproceedings{hsu2019learning,
  title={Learning-Based Frequency Estimation Algorithms.},
  author={Hsu, Chen-Yu and Indyk, Piotr and Katabi, Dina and Vakilian, Ali},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
## Table of contents
1. [Environment](#environment)
2. [Datasets](#datasets)
    1. [Internet traffic](#internet-traffic-dataset)
    1. [Search query](#search-query-dataset)
3. [Model training and inference](#model-training-and-inference)
4. [Using the model's predictions](#using-the-models-predictions)


## Environment

The code is developed under the following configurations:
- Ubuntu 14.04, Python 3.4
- Tensorflow-GPU 1.4.0, CUDA 8.0.61, cuDNN 6

To install required python packages, run ```pip install -r requirement.txt```.

For setting up tensorflow with GPU support (CUDA, cuDNN): https://www.tensorflow.org/install


## Datasets

### Internet traffic dataset

Dataset website: http://www.caida.org/data/passive/passive_dataset.xml

We preprocessed the number of packets and features for each unique internet flow in each minute (equinix-chicago.dirA.20160121-${minute}00.ports.npy):
```python
>>> import numpy as np
>>> data = np.load('equinix-chicago.dirA.20160121-140000.ports.npy').item()
>>> data.keys()
dict_keys(['y', 'x', 'note'])
>>> data['note']
'./data/2016/20160121-130000.UTC/equinix-chicago.dirA.20160121-140000'
>>> data['x'].shape         # 1142626 unique flows in this minute
(1142626, 11)
>>> data['x'][0][:8]        # source ip (first 4) and destination ip (last 4)
array([ 198.,  115.,   14.,  163.,    1.,   91.,  194.,    1.])
>>> data['x'][0][8:10]      # source port and destination port
array([     6.,  35059.])
>>> data['x'][0][-1]        # protocol type
22.0
>>> data['y'][0]            # number of packets
153733
```
Please request data access on the CAIDA website (https://www.caida.org/data/passive/passive_dataset_request.xml). We can share the preprocessed data once you email us the approval from CAIDA (usually takes 2~3 days).

### Search query dataset

Dataset website: https://jeffhuang.com/search_query_logs.html

We preprocessed the number of times a search query appears in each day in to separate files (aol_00${day}_len60.npz):
```python
>>> import numpy as np
>>> data = np.load('aol_0075_len60.npz')
>>> data.files
['query_lens', 'char_list', 'counts', 'queries', 'query_char_ids']
>>>
>>> data['queries'][0]          # the search query
'google'
>>> data['counts'][0]           # the number of times it appears
3158
>>> data['char_list']           # charector vocabulary
array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '.', '-',
       "'", '&', ';', 'nv'],
      dtype='<U2')
>>> data['query_char_ids'][0]   # the search query after character encoding (char -> int)
                                # queries longer than 60 characters are truncated.
array([  7.,  15.,  15.,   7.,  12.,   5.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.])
>>> data['query_lens'][0]       # length of the query
6
```
The preprocessed data can be downloaded here:
https://drive.google.com/open?id=1jWot63TJY8WHk5_bQPgIz-EPV0a6BDVX (~37GB after unpacking)

For evaluation (see "Using the model's predictions" below), download the (query, counts) pairs here:
https://drive.google.com/open?id=1RclH6mFvbGKm5aB4MQT9HVgQO94GLNmd (~30 GB after unpacking)


## Model training and inference

Example commands are in ```run.sh```.

```run_ip_model.py``` and ```run_aol_model.py``` include the model definitions, training, and inference code for internet traffic and search query datasets respectively. The models are defined in ```construct_graph()```. The code create multiple folders:
```
./log:          training config, git commit, terminal outputs for each run
./model:        saved model checkpoints (every ```--eval_n_epochs``` epochs)
./predictions   saved inference results (predicted counts for the items) during training
./summary:      tensorboard summary files
```
For only running model inference, add flag ```--evaluate``` and use ```--resume``` to provide the path to a model checkpoint.
To change the neural network, training, and other parameters, please check the input arguments.

## Using the model's predictions

To use the model's predictions in the LearnedSketch algorithm, see example commands in ```eval.sh```.

The script runs each algorithm with different amounts of space and different number of hash functions. We use the results to generate the loss vs space curves in the paper. Below shows the algorithms each python script implements:
```
count_min_param.py: Count-Min (CM)
cutoff_count_min_para.py: Learned CM, Table Lookup CM, Learned CM with Ideal Oracle
(use --count_sketch to replace Count-Min with Count-Sketch)
```
The results and logs will be saved in ```./param_results/```.

To reproduce the loss vs space curves in the papers using our trained model's predictions:
1. Download model predictions used in the paper here: https://drive.google.com/file/d/1PlmYUYEWHKJWOOyR1GrBuV3mcbSBdpiV/view
2. Run ```eval.sh``` (see comments for the different algorithms & baselines)
3. Run ```loss_vs_space_ip.sh``` (or ```loss_vs_space_aol.sh```) in the ```plotting``` folder to plot the figures

Alternatively, you can also download the saved ```param_results``` here: https://drive.google.com/open?id=1n2jDVhvKPwtFevyej42hGRieOeqIYoSJ
and run ```loss_vs_space_ip.sh``` (or ```loss_vs_space_aol.sh```) directly.

