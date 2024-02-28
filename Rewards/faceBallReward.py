from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

def faceBall():
    def reset(self, initial_state: GameState):
            pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

def faceBallReward():
    rewards_to_combine = (VelocityPlayerToBallReward(),
                            VelocityBallToGoalReward(),
                            faceBall(),
                            EventReward(team_goal=10, concede=-10, demo=0.1, touch=0.1, shot=0.4, save=0.5))
    reward_weights = (0.01, 0.1, 0.1, 10.0)

    return CombinedReward(reward_functions=rewards_to_combine,
                            reward_weights=reward_weights)


