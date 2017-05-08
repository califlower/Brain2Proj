# Emotion Recognition Using SNN

A series of Python programs for training an SNN to recognize emotions.

This was run on Ubuntu 16.04 LTS using Python 3.5.2.

This project was built off of the code for the paper "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" written by PU Diehl.

## Prerequisite

```
pip install -r requirements.txt
```

## Training The Network

1. Edit `Diehl&Cook_MNIST_random_conn_generator.py` and change lines 39-41 to suit your needs. `nInput` is the number of input neurons, `nE` is the number of excitatory neurons, and `nI` is the number of inhibitory neurons. Note that in our case `nE` = `nI` because each inhibitory neuron is connected to each excitatory neuron injectively (one-to-one).
2. Edit `Diehl&Cook_spiking_MNIST_Brian2.py` lines 221-235. You will need to change `num_examples` under `test_mode` to be the number of test images to present to the network. Under the `else` block directly under the check for `test_mode` you should change `num_examples` to reflect the number of training images. On lines 239-241 you should change the variables to reflect the chose values from step 1. `n_input` is the number of input neruons, `n_e` is the number of excitatory neurons, and `n_i` is the number of inhibitory neurons. On lines 447, 475, 245, 249, and 251 you should change `2043` to reflect the number of **test** images. On lines 449, 452, and 477 you should change `3499` to reflect the number of training images.
3. **Make sure `test_mode` on line 217 is `False`.**
3. Place your desired training data under `training/data1.csv`. The CSV should have columns [Category Code | 8-bit BW Pixel Values as Integers]. A new CSV will be created under `training/processed.csv`.
4. Run `Diehl&Cook_MNIST_random_conn_generator.py` and then `Diehl&Cook_spiking_MNIST_Brian2.py`.

## Testing the Network

Once your network has been trained you can do the following to test it:

1. Edit `Diehl&Cook_spiking_MNIST_Brian2.py` line 217 so that `test_mode` is `True`.
2. Place your desired testing data under `testing/data1.csv`.
3. Run `Diehl&Cook_spiking_MNIST_Brian2.py`.
4. Edit `Diehl&Cook_MNIST_evaluation.py` lines 87 and 88 so that `2043` is the number of testing images. Also, edit lines 94-95 where `n_e` is the number of excitatory neurons and `n_input` is the number of input neurons.
5. Run `Diehl&Cook_MNIST_evaluation.py`

## IMPORTANT NOTE

Before running `face_to_img.py` make sure that **both** `training/data1.csv` and `testing/data1.csv` exist.

In the event that you encounter numpy errors, it may be possible that you have run out of memory. Our testing was performed on an Ubuntu VM with 8 GB of RAM however for larger data sets you may need more memory.
