# Code amended from https://github.com/hsjeong5/MNIST-for-Numpy

import gzip
import os
import pickle
from urllib import request

import numpy as np


filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

base_path = 'data/MNIST'

def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for name in filename:        
        url = base_url+name[1]
        outfile = f'{base_path}/{name[1]}'
        if not os.path.exists(outfile):
            print("Downloading "+name[1]+"...")
            request.urlretrieve(url, outfile)

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(f'{base_path}/{name[1]}', 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(f'{base_path}/{name[1]}', 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(f"{base_path}/mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

cwd = os.getcwd()
os.chdir('../base')
os.makedirs(base_path, exist_ok=True)

download_mnist()
save_mnist()

os.chdir(cwd)