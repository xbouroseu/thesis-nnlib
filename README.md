# thesis-nnlib
## A library for building neural networks in nvidia-gpu accelerated systems

# Usage
## Requirements: NVIDIA GPU, Docker

## Step 1: Pull the docker image
```
$ docker pull sirmihawk/thesis:hpc22.11_build
```

## Step 2: Create a new docker container
The --rm option will create a container which will be auto-removed once the session is ended.

```
$ docker run -it --rm --gpus all sirmihawk/thesis:hpc22.11_build
```

## Step 3: Build the library and sample apps
Once you're inside the container you can clone again the repository with:
```
$ git clone https://github.com/xbouroseu/thesis-nnlib && cd thesis-nnlib
```
And then to build the library and apps you do:
```
$ make all
```

Alternatively you can only build the library with:
```
$ make lib
``` 

and the apps with:
```
$ make app
```

## Step 4: Run the mnist training application
After building the library and the sample apps we can run one application which is training and evaluating a Convolution Neural Network on the MNIST dataset.

Let's assume the variable `x` is the batch size:

```
$ cd apps/mnist_app
```

To run the gpu-accelerated version:
```
$ ./build/mnist_acc info x
```

And the non-accelerated one:
```
$ ./build/mnist_noacc info x
```