from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import RewardFunction
import numpy as np



"""
points for 
    shots
    touches 
    demo
    boost pickup
    boost amount
    face ball when touching (x cords)
    position defending 
    position attacking
        or basic RewardIfBehindBall from rlgym_sim.utils.reward_functions.common_rewards
    speed player
    speed toward ball
    distance to ball
    ball location (near oponnent goal)
    


punishment for 
    concede
    own goal 
    ball location (near own goal)
    existence
"""




class faceBall(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        euclid_pos_dif = np.linalg.norm(pos_diff)
        norm_pos_diff = pos_diff / euclid_pos_dif
        reward = float(np.dot(player.car_data.forward(), norm_pos_diff))
        if euclid_pos_dif < 10:
            return reward
        else:
            return 0

def lessFaceBallReward():
    rewards_to_combine = (VelocityPlayerToBallReward(),
                            VelocityBallToGoalReward(),
                            faceBall(),
                            EventReward(team_goal=10, concede=-10, demo=0.1, touch=0.1, shot=0.4, save=0.5))
    reward_weights = (0.01, 0.1, 0.01, 10.0)

    return CombinedReward(reward_functions=rewards_to_combine,
                            reward_weights=reward_weights)

