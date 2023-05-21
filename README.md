# Higher Distributed

This is the supporting code repository for the article

## Setup

(Optional) Setup a new Python environment via conda as:
```shell
conda env create -n <name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `1.13` on CUDA 11.8.
```shell
pip install 'torch<2' torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

Finally, in the same target environment (e.g. the one setup above), run to setup all the dependencies.
```shell
pip install -e .
```

## Run

We will use `CUDA_VISIBLE_DEVICES` environment variable to mask the number of GPUs available for use.

For instance, to use four GPUs:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_toy.py
```

The default parameters should not need changing for the demo.

**NOTE**: The device IDs may need to change as per hardware availability.

## License

MIT
