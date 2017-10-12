# dl_lsh

## Install
git clone https://github.com/TatianaJin/dl_lsh.git

## Build and Run
```
# in the home directory of your local copy
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Example execution script for mnist example:

```bash
max_epoch=20

hidden_layer_size=100
hidden_pool_size=10
num_hidden_layers=1

b=6
L=100
size_limit=0.5

L2_Lambda=0.003
learning_rate=0.01
num_threads=1

dropout=1
update_per_epoch=1

./MNISTExample ../data/train-labels.idx1-ubyte ../data/train-images.idx3-ubyte \
  ../data/t10k-labels.idx1-ubyte ../data/t10k-images.idx3-ubyte \
  $max_epoch $hidden_layer_size $hidden_pool_size $num_hidden_layers \
  $b $L $size_limit $L2_Lambda $learning_rate $num_threads $dropout $update_per_epoch
```
