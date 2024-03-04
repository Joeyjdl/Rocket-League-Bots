from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState
import numpy as np

class ExampleLogger(MetricsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.track_stats = [0]*10

    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,                                             #0
                game_state.players[0].car_data.rotation_mtx(),                                              #1

                # Stats to track per episode
                game_state.blue_score,                                                                      #2
                game_state.players[0].match_goals,                                                          #3
                game_state.orange_score,                                                                    #4
                0 if len(game_state.players) < 2 else game_state.players[1].match_goals,                    #5
                game_state.players[0].match_saves,                                                          #6
                0 if len(game_state.players) < 2 else game_state.players[1].match_saves,                    #7
                game_state.players[0].match_shots,                                                          #8
                0 if len(game_state.players) < 2 else game_state.players[1].match_shots,                    #9
                # end

                # avg
                game_state.players[0].boost_amount,                                                         #10
                0 if len(game_state.players) < 2 else game_state.players[1].boost_amount,                   #11
                game_state.players[0].car_data.position[2],                                                 #12
                0 if len(game_state.players) < 2 else game_state.players[1].car_data.position[2],           #13

                # max
                game_state.players[0].car_data.position[2],                                                 #14
                0 if len(game_state.players) < 2 else game_state.players[1].car_data.position[2],           #15
                ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        avg_stats = [0] * 4
        max_stats = [0] * 2


        for metric_array in collected_metrics:
            # [[velocity, rotation, score], [velocity, rotation, score], [velocity, rotation, score], [velocity, rotation, score]]
            avg_linvel += metric_array[0]
            current_stats = metric_array[2:10]

            for (idx, item) in enumerate(current_stats):
                if self.track_stats[idx] > 0 and item == 0:
                    self.track_stats[idx] = 0
                if item > self.track_stats[idx]:
                    self.track_stats[idx]=item 

            stats_to_avg = metric_array[10:14]
            for(idx, item) in enumerate(stats_to_avg):
                avg_stats[idx] += item

            stats_to_max = metric_array[14:16]
            for(idx, item) in enumerate(stats_to_max):
                if item > max_stats[idx]:
                    max_stats[idx] = item
            
        
        avg_linvel /= len(collected_metrics)
        for(idx, item) in enumerate(avg_stats):
                avg_stats[idx] /= len(collected_metrics)

        report = {
                    "x_vel":avg_linvel[0],
                    "y_vel":avg_linvel[1],
                    "z_vel":avg_linvel[2],
                    "Cumulative Timesteps":cumulative_timesteps,
                    "blue_team goals_scored": self.track_stats[0],
                    "blue_player goals_scored": self.track_stats[1],

                    "orange_team goals_scored": self.track_stats[2],
                    "orange_player goals_scored": self.track_stats[3],

                    "blue_player saves": self.track_stats[4],
                    "orange_player saves": self.track_stats[5],

                    "blue_player shots": self.track_stats[6],
                    "orange_player shots": self.track_stats[7],

                    "blue_player avg_boost": avg_stats[0],
                    "orange_player avg_boost": avg_stats[1],

                    "blue_player avg_height": avg_stats[2],
                    "orange_player avg_height": avg_stats[3],

                    "blue_player max_height": max_stats[0],
                    "orange_player max_height": max_stats[1],
                }
        wandb_run.log(report)
