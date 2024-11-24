# Dr. Strategy
[[website](https://ahn-ml.github.io/drstrategy/)] [[paper](https://arxiv.org/abs/2402.18866)]

This is the official code for the paper "Dr. Strategy: Model-Based Generalist Agents with Strategic Dreaming" (ICML 2024).


### Installation

```bash
./scripts_docker/build.sh

export WANDB_API_KEY=<your_wandb_api_key>

./scripts_docker/run.sh --device <gpu_id>
```


### Training
Available environments: `9rooms`, `25rooms`, `spiral9`, `mz7x7`, `mz15x15`, `robokitchen`, `dmc_walker`, `dmc_quad`

```bash

cd drstrategy

./runner/run_drstrategy_<env>.sh
```


### Evaluation
```bash
cd drstrategy

./eval_runner/eval_drstrategy_<env>.sh

```

#### Reference

```
@inproceedings{hamed2024drstrategy,
  title={Dr. Strategy: Model-Based Generalist Agents with Strategic Dreaming},
  author={Hamed, Hany and Kim, Subin and Kim, Dongyeong and Yoon, Jaesik and Ahn, Sungjin},
  booktitle={International Conference on Machine Learning},
  year = {2024},
}
```


### Acknowledgement

This codebase is based on the implementation of [Choreographer](https://skillchoreographer.github.io/).