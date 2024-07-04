# JaxNeuS

This is a [JAX](https://github.com/google/jax) implementation of [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://github.com/Totoro97/NeuS).

<p float="left">
    <img src="figures/scan114.gif" alt="drawing" width="200"/>
    <img src="figures/scan110.gif" alt="drawing" width="200"/>
    <img src="figures/bear.gif" alt="drawing" width="200"/>
    <img src="figures/thin.gif" alt="drawing" width="200"/>
</p>

## Installation

recommend using conda
```
conda env create --file conda.yaml
conda activate jaxneus
```

prepare [datasets](https://www.dropbox.com/scl/fo/um3wj3ctiuoottbfmqmgb/ABZRltszDvWHJ824UL6DHw0?rlkey=3vjok0aivnoiaf8z5j6w05k92&e=1&dl=0)
```
./download.sh
```


## Running

- **Training without masks**

```shell
python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name>
```

- **Training with masks**

```shell
python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name>
```

- **Extract surface from trained model** 

```shell
python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

- **View interpolation**

```shell
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding image set of view interpolation can be found in `exp/<case_name>/<exp_name>/render/`.

## Notes
Follow the NeuS code structure and command line for experiments. The rendering function is designed to handle a single ray and vectorized to apply to a batch of rays. The rendering procedure is optimized for speed using just-in-time compilation (JIT).

