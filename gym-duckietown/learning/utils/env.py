import gym
import gym_duckietown

def launch_env(id=None):
    env = None
    if id is None:
        # Launch the environment
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=3, # random seed
            #map_name="loop_empty",
            map_name="map2",
            max_steps=2001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=60, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env

