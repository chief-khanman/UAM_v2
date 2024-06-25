""" 
UAM Simulation Environment
--------------------------

The UAMSimEnv environment is a simulation of an urban air mobility (UAM) system. The environment is designed to be used for reinforcement learning research and development. The environment is implemented in Python and uses the OpenAI Gymnasium API.

This file registers the UAMSimEnv environment with OpenAI Gymnasium and exports the `gymnasium.make` function.

"""
__version__ = '0.0.1'
__all__ = ['UAMSim']  # In the future, add more environments here

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from gymnasium.envs.registration import register

try:
    for e in __all__:
        env_name = e[:-9]  # Limit the name to the first 9 characters
        register(
            id=f'{env_name}-v0',
            entry_point=f'uam_sim.envs:{env_name}',
            kwargs={},
        )
except:
    print("UAMSimEnv environment registration failed. Try harder.")
