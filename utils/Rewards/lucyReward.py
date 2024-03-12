from rlgym_sim.utils import math
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions import CombinedReward

from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward, AlignBallGoal, LiuDistancePlayerToBallReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import RewardFunction
import numpy as np


class BallToGoalDistance(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ballPos = state.ball.position
        ballSpeed = state.ball.linear_velocity
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc 
        wDisOff = 0.6
        wDisDef = 0.4
        distBallToGoal = ballPos - attacc
        depthGoal = ORANGE_GOAL_BACK - BACK_WALL_Y 
        normdistBallToGoalOff = np.linalg.norm(ballPos,attacc)
        normdistBallToGoalDef = np.linalg.norm(ballPos,protecc)
        wOff = np.exp(-0,5*(normdistBallToGoalOff- depthGoal)/6000*wDisOff)
        wDef = np.exp(-0,5*(normdistBallToGoalDef- depthGoal)/6000*wDisDef)
        return wOff - wDef


class BallToGoalVelocity(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ballPos = state.ball.position
        ballSpeed = state.ball.linear_velocity
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc        
        
        distBallToGoal = ballPos - attacc
        normdistBallToGoal = np.linalg.norm(ballPos,attacc)

        return float((distBallToGoal/normdistBallToGoal)*(ballSpeed/6000))


class SaveBoost(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.sqrt(state.players[0].boost_amount/100)


class DistWeightedAlignment(RewardFunction):
    def reset(self, initial_state: GameState):
        pass
    
    def sgn(array):
        for num in array:
            if num <= 0:
                return -1
        return 1

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        alignBallGoalReward = AlignBallGoal().get_reward(player, state, previous_action)
        distToBallReward = LiuDistancePlayerToBallReward().get_reward(player,state, previous_action)
        krcSgn = self.sgn(np.array([alignBallGoalReward, distToBallReward]))

        return np.linalg.norm(alignBallGoalReward*distToBallReward)**0.5 * krcSgn
    

class OffPotential(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def sgn(array):
        for num in array:
            if num <= 0:
                return -1
        return 1
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        alignBallGoalReward = AlignBallGoal().get_reward(player, state, previous_action)
        distToBallReward = LiuDistancePlayerToBallReward().get_reward(player, state, previous_action)
        playerToBallVelocityReward = VelocityPlayerToBallReward.get_reward(player, state, previous_action)
        krcSgn = self.sgn(np.array([alignBallGoalReward, distToBallReward, playerToBallVelocityReward]))
        return np.linalg.norm(alignBallGoalReward*distToBallReward*playerToBallVelocityReward)**(1/3) * krcSgn


class touchBallToGoal(RewardFunction):
    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        player = initial_state.players[0]
        self.lastRegisteredBallsTouched = player.ball_touched
        self.lastRegisteredballSpeedToGoal = BallToGoalVelocity.get_reward(player, initial_state)
    
    def __init__(self):
        super().__init__()
        # Need to keep track of last registered value to detect changes
        self.lastRegisteredBallsTouched = 0
        self.lastRegisteredballSpeedToGoal = 0
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_value = self.lastRegisteredBallsTouched
        new_value = player.ball_touched

        ballSpeedToGoal = BallToGoalVelocity.get_reward(player, state)

        diff_value = new_value - old_value
        if diff_value > 0:
            reward = 1 * (ballSpeedToGoal - self.lastRegisteredballSpeedToGoal)

        self.lastRegisteredBallsTouched = new_value
        self.lastRegisteredballSpeedToGoal = ballSpeedToGoal

        return reward
    


def lucyReward():
    rewards_to_combine = (BallToGoalDistance(),
                          BallToGoalVelocity(),
                          SaveBoost(),
                          DistWeightedAlignment(),
                          OffPotential(),
                          touchBallToGoal(),
                          EventReward(team_goal=10, concede=-3, shot=1.5, touch=0.05))
    reward_weights = (2, 0.8, 0.5, 0.6, 1, 0.25, 1)

    return CombinedReward(reward_functions=rewards_to_combine,
                            reward_weights=reward_weights)



