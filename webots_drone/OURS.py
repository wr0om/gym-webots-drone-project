import gym
import time
env = gym.make('webots_drone:webots_drone/DroneEnvDiscrete-v0',
               time_limit_seconds=60,  # 1 min
               max_no_action_seconds=5,  # 30 sec
               frame_skip=25,  # 200 ms # 125,  # 1 sec
               goal_threshold=25.,
               init_altitude=25.,
               altitude_limits=[11, 75],
               is_pixels=False)
print('reset')
env.reset()
for _ in range(1250):
    env.render()
    action = env.action_space.sample()
    print('action', action)
    env.step(action) # take a random actionenv.close()