import random
from math import exp,isfinite
from typing import Callable, Final
from pycommons.types import type_error
from moptipy.utils.strings import num_to_str_for_name
from moptipy.algorithms.modules.temperature_schedule import ExponentialSchedule
from moptipy.algorithms.modules.temperature_schedule import TemperatureSchedule

class ExponentialSchedule(TemperatureSchedule):

    def __init__(self, t0: float, epsilon: float) -> None:
    
        super().__init__(t0)

        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (not (0.0 < epsilon < 1.0)):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be in (0,1).")

        #: the epsilon parameter of the exponential schedule
        self.epsilon: Final[float] = epsilon
        #: the value used as basis for the exponent
        self.__one_minus_epsilon: Final[float] = 1.0 - epsilon

        if not (0.0 < self.__one_minus_epsilon < 1.0):
            raise ValueError(
                f"epsilon cannot be {epsilon}, because 1-epsilon must be in "
                f"(0, 1) but is {self.__one_minus_epsilon}.")


    def temperature(self, tau: int) -> float:

        return self.t0 * (self.__one_minus_epsilon ** tau)

    def __str__(self) -> str:
        
            return (f"exp{num_to_str_for_name(self.t0)}_"
                    f"{num_to_str_for_name(self.epsilon)}")

def calc_next_step(best_hyperparameter:float) -> float:
    new_hyperparameter = best_hyperparameter + random.uniform(-0.025,0.025) 
    return new_hyperparameter

def choose_model(schedule:ExponentialSchedule,best:float, tau:int, new:float)->bool:
    temperature: Final[Callable[[int], float]] = schedule.temperature
    r01: Final[Callable[[], float]] = random.random  # random from [0, 1]
    if (new <= best) or (  # Accept if <= or if SA criterion
        r01() < exp((best - new) / temperature(tau))):
            return True
    return False

next_step = calc_next_step(0.2)
print(next_step)

schedule = ExponentialSchedule(10.0,0.05)

step_accepted = choose_model(schedule,99,1,100)
print(step_accepted)
