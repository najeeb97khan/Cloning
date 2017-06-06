## Requirements for running the code
Make sure that both python and python install package manager (pip) is up and running. 
Python v2.7 can be installed from here https://www.python.org/downloads/
To install pip use the following command on a linux distro
>sudo easy_install --upgrade pip

1. Python v2.7
2. Tensorflow v1.1
3. Numpy
4. matplotlib (optional)
4. tqdm (visualising the training)

## Running the code
>pip install -r requirements.txt
>python black_box.py

black_box.py will create the Black Box network. A checkpoint for the network is already stored in the checkpoints/black_box file. If you wish to train it more on the MNIST dataset then answer the query with **y**, otherwise answer with **n**. The code will generate a file (of large size) containing a random dataset and their corresponding labels

>python white_box.py

This code segment will construct a new object for the black box and try to learn the parameters using the dataset generated above. Since the gradients tend to explode, the program will result in an error. To remove the error go to **black_box.py**, line 66, 134 and follow the instructions. This will allow the white box to be trained but with explosive gradients hence leading to **nan** loss.

## Running the visualisation notebook
An Ipython notebook visualising the random data is also present. Install **anaconda** data science platform to run the ipython notebook from here https://www.continuum.io/downloads
Tor run the ipython notebook use the following command
>jupyter notebook

The notebook will open in any of the default web browsers.
