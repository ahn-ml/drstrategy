#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import drstrategy_envs.miniworld

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--continuous', action='store_true', help='discrete or continuous, default is discrete')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--obs_level',default=2, help='observation level. 1 for POMDP-2D and 2 for MDP-2D, 3 for POMDP-3D')
parser.add_argument('--agent_mode', default='circle')
parser.add_argument('--room_size', default=5)
args = parser.parse_args()

env = gym.make(args.env_name, obs_level=int(args.obs_level), continuous=args.continuous, agent_mode=args.agent_mode, room_size=int(args.room_size))

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

env.reset()

# Create the display window
env.render('pyglet', view=view_mode)

def step(action):
    if env.continuous:
        print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, action))
    else:
        print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    if env.continuous:
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            env.reset()
            env.render('pyglet', view=view_mode)
            return
        
        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
        
        if symbol == key.P:
            speed, deg = input("Enter two values: ").split()
            print(speed, deg)
            speed, deg = float(speed), float(deg)
            velo = np.array([speed, deg])
            step(action=velo)
    else:
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            env.reset()
            env.render('pyglet', view=view_mode)
            return
        
        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
        
        if symbol == key.UP:
            step(env.actions.move_forward)
        elif symbol == key.DOWN:
            step(env.actions.move_back)
        elif symbol == key.LEFT:
            step(env.actions.turn_left)
        elif symbol == key.RIGHT:
            step(env.actions.turn_right)
        elif symbol == key.PAGEUP or symbol == key.P:
            step(env.actions.pickup)
        elif symbol == key.PAGEDOWN or symbol == key.D:
            step(env.actions.drop)
        elif symbol == key.ENTER:
            step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()
