# SMA4DM2S_NN
A Neural Network-based Sequence Memory Agent for a Delayed Match-to-Sample Task

## How to Install
* Clone the repository

* Clone [SequenceMemory](https://github.com/rondelion/SequenceMemory)

* Install numpy and [gymnasium](https://gymnasium.farama.org)

* Register the environment to Gym
    * Place `M2S_Env2.py` file in `gym/gym/envs/myenv`  
    (wherever Gym to be used is installed)
    * Add to `__init__.py` (located in the same folder)  
      `from gym.envs.myenv.M2S_Env2 import M2S_Env2`
    * Add to `gym/gym/envs/__init__.py`  
```
register(
    id='M2S_Env2-v0',
    entry_point='gym.envs.myenv:M2S_Env2'
    )
```

## Usage
### Command arguments
- Options
      --dump: dump file path
      --episode_count: Number of training episodes (default: 1)
      --episode_sets: Number of episode sets (default: 1)
      --eval_period: Evaluation span (default: 5)
      --config: Model configuration (default: RuleFinder1.json)
      --dump_level >= 0

### Sample usage
```
$ python RuleFinder1.py --episode_count 100 --eval_period 20 --episode_sets 1 --dump log.txt --dump_level 2

```
