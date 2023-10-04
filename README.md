# cgp-cnn-PyTorch-Updated
A Genetic Programming Approach to Designing CNN Architectures, In GECCO 2017 (oral presentation, Best Paper Award).
This repository is a fork of the official implementation by Masanori Suganuma, and has been updated to be compatible with a more recent version of Pytorch and CUDA.

## Personal Implementation Requirements for Function
* Python 3.6.8
* Pandas 1.1.5
* Scikit-Image 0.17.2
* Pytorch 1.10.0
* CUDA Version 10.2

The Python package requirements can be installed with the command:
```shell
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==1.1.5 scikit-image==0.17.2
```

# Designing Convolutional Neural Network Architectures Based on Cartegian Genetic Programming

This repository contains the code for the following paper:

Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures," 
Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17, Best paper award), pp. 497-504 (2017) [[paper]](https://doi.org/10.1145/3071178.3071229) [[arXiv]](https://arxiv.org/abs/1704.00764)

## Usage

### Run the architecture search
This code can reproduce the experiment for CIFAR-10 dataset with the same setting of the GECCO 2017 paper (by default scenario). The (training) data are split into the training and validation data. The validation data are used for assigning the fitness to the generated architectures.

When you use the multiple GPUs, please specify the `-g` option:

```shell
python exp_main.py -g 2
```

After the execution, the files, `network_info.pickle` and `log_cgp.txt` will be generated. The file `network_info.pickle` contains the information for Cartegian genetic programming (CGP) and `log_cgp.txt` contains the log of the optimization and discovered CNN architecture's genotype lists.

Some parameters (e.g., # rows and columns of CGP, and # epochs) can easily change by modifying the arguments in the script `exp_main.py`.

### Re-training

The discovered architecture is re-trained by the different training scheme (500 epoch training with momentum SGD) to polish up the network parameters. All training data are used for re-training, and the accuracy for the test data set is reported.

```shell
python exp_main.py -m retrain
```
