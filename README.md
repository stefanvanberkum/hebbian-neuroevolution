# Hebbian neuroevolution

Evolutionary architecture search for Hebbian deep learning.

## Usage

The following commands can be used to run all the experiments.

#### Evolution

The following will start a new run and place the results in ```runs/$RUN```, where ```$RUN``` is a name generated upon
running based on the current date and time.

```
python evolution.py
```

Arguments:

- --dataset: The dataset to use for evolution (default: CIFAR10)
- --n_channels: The initial number of channels (default: 8).
- --scaling_factor: The scaling factor for the number of filters after each reduction cell (default: 4).
- --n_ops: The number of operations in each cell (default: 5).
- --stack_size: The normal cell stack size (default: 0).
- --n_reduction: The number of reduction cells, used if ``stack_size`` is set to zero (default: 3).
- --n_epochs: The epoch increment for training the classifier (default: 10).
- --generations: The number of generations (default 100).
- --eta: The base learning rate used in SoftHebb convolutions (default: 0.01).
- --encoder_batch: The batch size for training the encoder's SoftHebb convolutions (default: 32).
- --classifier_batch: The batch size for training the classifier with SGD (default: 256).
- --reduce: Turn on this option to reduce the spatial dimension of the input images to 16x16.
- --verbose: Turn on this option for verbose info.
- --checkpoint: Optional checkpoint name to continue evolution.

#### Tuning

```
# Retrain the best five models from evolution.
python tuning.py --run $RUN

# Tune the hyperparameters of HebbNet-A.
python tuning.py --run HebbNet --tune

# Tune the hyperparameters of the original SoftHebb network.
python tuning.py --run SoftHebb --tune
```

Arguments:

- --run: The evolution run to load or model to tune (HebbNet or SoftHebb in that case).
- --n_channels: The number of initial channels.
- --cv: Turn on this option to cross-validate the best architectures from the specified run.
- --tune: Turn on this option to tune HebbNet-A or the SoftHebb network.
- --ray: Turn on this option to tune using Ray Tune.

#### Experiments

```
# Run the classical analysis for any model, mode, and dataset.
python experiments.py --model $model --mode $mode --$dataset

# Summarize results.
python experiments.py --summarize

# Run the Bayesian analysis.
python experiments.py --bayesian
```

Arguments:

- --model: The model to test, one of: {HebbNet, SoftHebb}.
- --mode: The training mode to be applied, one of: {Hebbian, BP}.
- --CIFAR10: Turn on this option to test on CIFAR-10.
- --CIFAR100: Turn on this option to test on CIFAR-100.
- --SVHN: Turn on this option to test on SVHN.
- --summarize: Turn on this option to summarize results.
- --visualize: Turn on this option to visualize results.
- --bayesian: Turn on this option to run a Bayesian analysis.

## File description

- ```activations.py```
    - Provides activation functions.
- ```architecture.py```
    - Provides architecture and cell objects used in evolutionary architecture search.
- ```dataloader.py```
    - Provides methods for loading and preprocessing data.
- ```evolution.py```
    - Provides the method for evolutionary architecture search.
- ```experiments.py```
    - Provides methods for running the experiments after evolution.
- ```layercam.py```
    - Provides a LayerCAM implementation based on pytorch-grad-cam.
- ```layers.py```
    - Provides PyTorch layers.
- ```models.py```
    - Provides models.
- ```training.py```
    - Provides methods for model training.
- ```tuning.py```
    - Provides methods for hyperparameter tuning after evolution.
