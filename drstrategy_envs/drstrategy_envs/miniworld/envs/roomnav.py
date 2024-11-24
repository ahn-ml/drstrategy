import random
import numpy as np
import math

from drstrategy_envs.miniworld.params import DEFAULT_PARAMS
from ..miniworld import CustomMiniWorldEnv
from ..entity import Box
from ..entity import *
from gym import spaces



class OneRoom(CustomMiniWorldEnv):
    """

    Traverse a room
    -----
    | 0 |
    -----

    """

    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = []
        else:
            assert len(connections) == 0, "Connection should be exactly 0"
            self.connections = connections
        
        if textures is None:
            self.textures = ['lightgray']
        else:
            assert len(textures) == 1, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert (self.placed_room == 0), "placing point should be in 0"
            self.placed_room = placed_room

        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size


        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(1):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for k in range(9):
            self.place_entity(
                ent=Box(list(COLORS.keys())[k+1], transparentable=True, size=2*self.room_size/15),
                pos=[(self.room_size/3)*(k%3)+0.16*self.room_size, 0, (self.room_size/3)*(k//3)+0.16*self.room_size],
                dir=0)
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info


class TwoRoomsVer1(CustomMiniWorldEnv):
    """

    Traverse the 2 rooms

    ---------
    | 0 | 1 |
    ---------

    """

    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['lightbeige', 'lightgray']
        else:
            assert len(textures) == 2, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 2)), "placing point should be in 0~1"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(2):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(1):
            for j in range(2):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
    

class TwoRoomsVer0(CustomMiniWorldEnv):
    """

    Traverse the 2 rooms

    ---------
    | 0 | 1 |
    ---------

    """

    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2, 
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige']
        else:
            assert len(textures) == 2, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 2)), "placing point should be in 0~1"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(2):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
                
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(1):
            for j in range(2):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info


class ThreeRooms(CustomMiniWorldEnv):
    """

    Traverse the 3 rooms

    -------------
    | 0 | 1 | 2 |
    -------------

    """
    
    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (1,2)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige', 'lightgray']
        else:
            assert len(textures) == 3, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 3)), "placing point should be in 0~2"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(3):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(1):
            for j in range(3):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
    
class FiveRooms(CustomMiniWorldEnv):
    """

    Traverse the 5 rooms

    ---------------------
    | 0 | 1 | 2 | 3 | 4 |
    ---------------------

    """
    
    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (1,2), (2, 3), (3, 4)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue']
        else:
            assert len(textures) == 5, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 5)), "placing point should be in 0~2"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(5):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            _start_x = self.room_size*2
            self.place_agent(pos=[_start_x+ self.room_size/2, 0, self.room_size/2])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(1):
            for j in range(5):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info

class StraightTwentyFiveRooms(CustomMiniWorldEnv):
    """

    Traverse the 25 rooms in one line starting from the middle

    --------------------------------------------------------------------------------------------------------------------
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
    --------------------------------------------------------------------------------------------------------------------

    """
    
    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (1,2), (2, 3), (3, 4), (4, 5), (5, 6), (6,7), (7,8), (8,9),
                                (9,10), (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
                                (18,19),(19,20),(20,21),(21,22),(22,23),(23,24),]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['crimson','beanpaste', 'cobaltgreen', 'lightnavyblue', 'skyblue', 
                             'lightcobaltgreen','oakbrown', 'copperred', 'lightgray', 'lime',
                             'turquoise', 'violet', 'beige', 'morningglory', 'silver',
                             'magenta','sunnyyellow', 'blueberry', 'lightbeige', 'seablue',
                             'lemongrass', 'orchid', 'redbean', 'orange', 'realblueberry']
        else:
            assert len(textures) == 25, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 25)), "placing point should be in 0~2"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for i in range(1):
            for j in range(25):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            _start_x = self.room_size*12
            self.place_agent(pos=[_start_x+ self.room_size/2, 0, self.room_size/2])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(1):
            for j in range(25):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info


class NineRooms(CustomMiniWorldEnv):
    """
    
    Traverse the 9 rooms
    
    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |
    ------------- 

    """


    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,8), (6,7), (7,8)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue', 'lightcobaltgreen',
                             'oakbrown', 'navyblue', 'cobaltgreen']
        else:
            assert len(textures) == 9, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 9)), "placing point should be in 0~8"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size
        
        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        
        rooms = []
        for i in range(3):
            for j in range(3):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(3):
            for j in range(3):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info
    

class SpiralNineRooms(CustomMiniWorldEnv):
    """
    
    Traverse the 9 rooms in spiral
    
    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |
    ------------- 

    """


    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue', 'lightcobaltgreen',
                             'oakbrown', 'navyblue', 'cobaltgreen']
        else:
            assert len(textures) == 9, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 9)), "placing point should be in 0~8"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size
        
        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        
        rooms = []
        for i in range(3):
            for j in range(3):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(3):
            for j in range(3):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info
    


class TwentyFiveRooms(CustomMiniWorldEnv):
    """
    
    Traverse the 25 rooms
    
    ---------------------
    | 0 | 1 | 2 | 3 | 4 | 
    ---------------------
    | 5 | 6 | 7 | 8 | 9 |
    ---------------------
    |10 |11 |12 |13 |14 |
    ---------------------
    |15 |16 |17 |18 |19 |
    ---------------------
    |20 |21 |22 |23 |24 |
    ---------------------

    """


    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9),
                                (5,6), (5,10), (6,7), (6,11), (7,8), (7,12), (8,9), (8,13), (9,14),
                                (10,11), (10,15), (11,12), (11,16), (12,13), (12,17), (13,14), (13,18), (14,19),
                                (15,16), (15,20), (16,17), (16,21), (17,18), (17,22), (18,19), (18,23), (19,24),
                                (20,21), (21,22), (22,23), (23,24)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['crimson','beanpaste', 'cobaltgreen', 'lightnavyblue', 'skyblue', 
                             'lightcobaltgreen','oakbrown', 'copperred', 'lightgray', 'lime',
                             'turquoise', 'violet', 'beige', 'morningglory', 'silver',
                             'magenta','sunnyyellow', 'blueberry', 'lightbeige', 'seablue',
                             'lemongrass', 'orchid', 'redbean', 'orange', 'realblueberry']
        else:
            assert len(textures) == 25, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 25)), "placing point should be in 0~24"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size
        
        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        
        rooms = []
        for i in range(5):
            for j in range(5):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[5*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(5):
            for j in range(5):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info

class SpiralTwentyFiveRooms(CustomMiniWorldEnv):
    """
    
    Traverse the 25 rooms in spiral
    
    ---------------------
    | 0 | 1 | 2 | 3 | 4 | 
    ---------------------
    | 5 | 6 | 7 | 8 | 9 |
    ---------------------
    |10 |11 |12 |13 |14 |
    ---------------------
    |15 |16 |17 |18 |19 |
    ---------------------
    |20 |21 |22 |23 |24 |
    ---------------------

    """


    def __init__(self, connections: list=None, textures: list=None, placed_room=None, obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(12,13), (8, 13), (7,8), (6, 7), (6,11), (11,16), (16,17),
                                (17,18), (18,19), (14, 19), (9, 14), (4, 9), (3,4), (2,3),
                                (1,2), (0,1), (0,5), (5,10), (10,15), (15,20), (20,21), (21,22),
                                (22,23), (23,24)]
                
                # (0,1), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9),
                #                 (5,6), (5,10), (6,7), (6,11), (7,8), (7,12), (8,9), (8,13), (9,14),
                #                 (10,11), (10,15), (11,12), (11,16), (12,13), (12,17), (13,14), (13,18), (14,19),
                #                 (15,16), (15,20), (16,17), (16,21), (17,18), (17,22), (18,19), (18,23), (19,24),
                #                 (20,21), (21,22), (22,23), (23,24)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['crimson','beanpaste', 'cobaltgreen', 'lightnavyblue', 'skyblue', 
                             'lightcobaltgreen','oakbrown', 'copperred', 'lightgray', 'lime',
                             'turquoise', 'violet', 'beige', 'morningglory', 'silver',
                             'magenta','sunnyyellow', 'blueberry', 'lightbeige', 'seablue',
                             'lemongrass', 'orchid', 'redbean', 'orange', 'realblueberry']
        else:
            assert len(textures) == 25, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 25)), "placing point should be in 0~24"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size
        
        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        
        rooms = []
        for i in range(5):
            for j in range(5):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[5*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)

            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        
        if pos is None:
            _start_x = self.room_size*4
            self.place_agent(pos=[_start_x+self.room_size-2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(5):
            for j in range(5):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )
            
    

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info
