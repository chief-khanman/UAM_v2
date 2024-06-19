# Register the UAMSim environment
from gym.envs.registration import register

register(
    id='uam_sim-v0',
    entry_point='uam_sim.envs:UAMSimEnv',
)