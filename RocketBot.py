import rlgym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.callbacks import CheckpointCallback

# from stable_baselines3.common.save_util import save_to_zip_file
# from stable_baselines3.common.save_util import load_from_zip_file

import os

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, model):
        super(SaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.model = model

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            PPO.save(self.model, path=self.save_path)
        return True


print(f"Code has begun")
#Make the default rlgym environment
env = rlgym.make()

#Initialize PPO from stable_baselines3
print(f"Loading has begun")
if os.path.exists('./BotBrains/RockBotBrain.zip'):
    model = PPO.load('./BotBrains/RockBotBrain', env=env, verbose=1)
else:
    model = PPO("MlpPolicy", env=env, verbose=1)
# model = None
# if os.path.exists("RockBotBrain.zip"):
#     model = load_from_zip_file("RockBotBrain")
# if not isinstance(model, PPO):
#     model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
# Set up the callback to save the model every `save_freq` steps
save_freq = 100  # Adjust as needed
save_path = "./BotBrains/RockBotBrain"
save_callback = SaveModelCallback(save_freq, save_path, model)
# checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)

# Train the model with the callback
print(f"Training has begun")
model.learn(total_timesteps=int(1e6), callback=save_callback)
