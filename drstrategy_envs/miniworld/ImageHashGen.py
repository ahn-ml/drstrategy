import gym
import json
from PIL import Image
import imagehash
import drstrategy_envs.miniworld
from gym.wrappers import ResizeObservation


task = 'NineRooms'
env = ResizeObservation(gym.make(f'MiniWorld-{task}-v0', obs_level=1, continuous=True, room_size=15, agent_mode='empty'), shape=64)

traj_obs = {}

task_iter_dict = {'OneRoom':[1,1], 'TwoRoomsVer0':[2,1], 'TwoRoomsVer1': [2,1], 
                  'ThreeRooms':[3,1], 'NineRooms':[3,3]}

for i in range(task_iter_dict[task][0]):
    for j in range(task_iter_dict[task][1]):
        for k in range(14):
            obs = env.reset(pos=[15*i+0.5, 15*j+k+0.5])
            traj_obs[str(imagehash.phash(Image.fromarray(obs)))] = [15*i+0.5, 15*j+k+0.5]
            
            for l in range(1,14):
                obs, rew, done, info = env.step([1, 0])
                traj_obs[str(imagehash.phash(Image.fromarray(obs)))] = [15*i+l+0.5, 15*j+k+0.5]
                


json.dump(traj_obs, open(f'imghash_{task}_15.json', 'w'))


