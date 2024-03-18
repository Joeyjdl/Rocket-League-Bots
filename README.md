# Rocket-League-Bots
A 1v0 bot trained using Rlgym_ppo

## Training
1. Clone this git repo using `git clone https://github.com/Joeyjdl/Rocket-League-Bots.git`
2. Install requirements with: `pip install -r requirements.txt`
3. The reward function can be changed in `utils/envBuilder/build_rocketsim_env.py`
4. You can then start training by running `python Training.py <run_number>` where `<run_number>` is used for logging purposes and future reference

When running for the first time, the script may ask you for a wandb API key, if you are not interested in the logs, then you can choose to disable them

## Performance
There are several options for looking at the bots performance
1. Wandb logs
2. RLviser
3. RLbot

### Wandb logs
These logs can be found at https://wandb.ai, the most notable statistics are:
- Goal_rate
- Policy_reward
- Overal steps per second

### RLviser
Mostly useful to confirm the bot isn't exploiting your reward function. Can be run with `python Rlviser.py <run_number>`, or alternatively `python Rlviser.py <reward_func>_<run_number>`

You can append `--count` to the run command to purely look at the amount of goals it scores. The amount of episodes it runs for can be changed in the `Rlviser.py` file

### RLbot
If you want to evaluate performance by playing against the bot (or having it play against itself / another bot) then use RLbot.
1. Copy `PPO_POLICY.pt` from `data/checkpoints/<bot>/<checkpoint>` to `rlbot/src/Brain`
2. Add the folder to RLbot by pressing the button in the top left
3. Allow RLbot to install the dependencies
4. Play!


## Required packages
- Numpy
- Rlgym_ppo
- Rocketsim
- Rlgym_sim

Rocket league and RLbot are required if you want to play against the bot



