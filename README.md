# Towards Deeper Understanding of PPR-based Embedding Approaches: A Topological Perspective (PPREI)

This is our Pytorch implementation for the paper:

> Xingyi Zhang, Zixuan Weng, Sibo Wang. Towards Deeper Understanding of PPR-based Embedding Approaches: A Topological Perspective." WWW 2024.


## Requirements:
* Python=3.10.12
* PyTorch=1.12.0
* Scipy=1.10.1
* networkx=2.8.4
* numpy=1.24.3
## Preprocessing the dataset:
run processData

**Parameters**

- f: the name of dataset
- r: the rank of matrix 

```
python processData.py -f Brazil
```
## Training:

run PPR

**Parameters**

- f: the name of dataset
- r: the rank of matrix 
- a: the value of teleport probability
- i: the number of training epoch
- t: the step of PPR 
- e: the value of threshould

```
python  PPR.py -f Brazil.mat -r 128 -a 0.7 -i 40 -t 10 -e 1e-7
```

run network_stats_PPR

**Parameters**

- f: the name of dataset
- r: the rank of matrix 

```
python network_stats_PPR.py -f Brazil.mat -r 128
```
**Download datasets**
https://drive.google.com/drive/folders/1zsCFA7U8ZKV1bg5g9IwNxqf1TzgGwokw?usp=sharing
## Acknowledgements:
Part of the code refers to https://github.com/konsotirop/Invert_Embeddings, which is the official implementation of the paper:
> Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis. "DeepWalking Backwards: From Embeddings Back to Graphs." ICML 2021.
## Reference:
Any scientific publications that use our codes should cite the following paper as the reference:

```
 @inproceedings{zhang&weng2024PPREI,
     title = "Towards Deeper Understanding of PPR-based Embedding Approaches: A Topological Perspective",
     author = {Xingyi Zhang and
     		  Zixuan Weng and
     		  Sibo wang},
     booktitle = {{WWW}},
     year = {2024},
 }
 ```

If you have any questions for our paper or codes, please send an email to zxweng0701@gmail.com.
