import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

HabitatSimActions.extend_action_space("NOISY_MOVE_FORWARD")
HabitatSimActions.extend_action_space("NOISY_TURN_RIGHT")
HabitatSimActions.extend_action_space("NOISY_TURN_LEFT")

from .pyrobot_noisy_controls import *
