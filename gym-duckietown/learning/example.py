import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from reinforcement.pytorch.ddpg import DDPG
# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=2000, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map1')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

env = DuckietownEnv(
    map_name = args.map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = args.max_steps,
    seed = args.seed
)


state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])   
max_action = float(0.8) # vel and angel limit to 0.8.
print("ddpg param")
print(state_dim)
print(action_dim)
print(max_action)

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
policy.load(filename='ddpg', directory='reinforcement/pytorch/models/')

obs = env.reset()
env.render()

total_reward = 0

# please remove this line for your own policy
#actions = np.loadtxt('./map5_seed11.txt', delimiter=',')
#actions = np.loadtxt('./control_v_a.txt', delimiter=',')
done = False
actions = []
while True:
    while not done:
        action = policy.predict(np.array(obs))
            # Perform action
            #print(action[0])
            #print(action[1])

            #action_v_a = wheel2velangle(action)
            #actions.append(action_v_a)

        obs, reward, done, info = env.step(action)
        print("reward is {}".format(reward))
        env.render()
    done = False
    obs = env.reset()
    #np.savetxt('./control_v_a.txt', actions, delimiter=',')



"""
for (speed, steering) in actions:

    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    
    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

print("Total Reward", total_reward)
"""

# dump the controls using numpy
#np.savetxt('./map5_seed11.txt', actions, delimiter=',')