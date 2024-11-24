import numpy as np
from scipy.spatial.transform import Rotation

def shortest_angle(angle):
  if not angle.shape:
    return shortest_angle(angle[None])[0]
  angle = angle % (2 * np.pi)
  angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
  return angle

def quat2euler(quat):
  rot = Rotation.from_quat(quat)
  return rot.as_euler('XYZ')

def get_dmc_benchmark_goals(task_type):
  if task_type == 'walker':
    # pose[0] is height
    # pose[1] is x
    # pose[2] is global rotation
    # pose[3:6] - first leg hip, knee, ankle
    # pose[6:9] - second leg hip, knee, ankle
    # Note: seems like walker can't bend legs backwards
    
    lie_back = [[ -1.2 ,  0. ,  -1.57,  0, 0. , 0.0, 0, -0.,  0.0]]
    lie_front = [[-1.2, -0, 1.57, 0, 0, 0, 0, 0., 0.]]
    legs_up = [[ -1.24 ,  0. ,  -1.57,  1.57, 0. , 0.0,  1.57, -0.,  0.0]]

    kneel = [[ -0.5 ,  0. ,  0,  0, -1.57, -0.8,  1.57, -1.57,  0.0]]
    side_angle = [[ -0.3 ,  0. ,  0.9,  0, 0, -0.7,  1.87, -1.07,  0.0]]
    stand_up = [[-0.15, 0., 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1]]
    
    lean_back = [[-0.27, 0., -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]]
    boat = [[ -1.04 ,  0. ,  -0.8,  1.6, 0. , 0.0, 1.6, -0.,  0.0]]
    bridge = [[-1.1, 0., -2.2, -0.3, -1.5, 0., -0.3, -0.8, -0.4]]

    head_stand = [[-1, 0., -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]]
    one_feet = [[-0.2, 0., 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]]
    arabesque = [[-0.34, 0., 1.57, 1.57, 0, 0., 0, -0., 0.]]
    # Other ideas: flamingo (hard), warrior (med), upside down boat (med), three legged dog

    goals = np.stack([lie_back, lie_front, legs_up, 
                      kneel, side_angle, stand_up, lean_back, boat,
                      bridge, one_feet, head_stand, arabesque])

  if task_type == 'quadruped':
    # pose[0,1] is x,y
    # pose[2] is height
    # pose[3:7] are vertical rotations in the form of a quaternion (i think?)
    # pose[7:11] are yaw pitch knee ankle for the front left leg
    # pose[11:15] same for the front right leg
    # pose[15:19] same for the back right leg
    # pose[19:23] same for the back left leg

    
    lie_legs_together = [get_quadruped_pose([0, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7])]
    lie_rotated = [get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]))]
    lie_two_legs_up = [get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[1, 3], down=[0, 2]))]

    lie_side = [get_quadruped_pose([0., 0, -1.57], 0.3, dict(out=[0,1,2, 3]), [-0.7, 0.7, -0.7, 0.7])]
    lie_side_back = [get_quadruped_pose([0., 0, 1.57], 0.3, dict(out=[0,1,2, 3]), [-0.7, 0.7, -0.7, 0.7])]
    stand = [get_quadruped_pose([1.57, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))]
    stand_rotated = [get_quadruped_pose([0.8, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))]
 
    stand_leg_up = [get_quadruped_pose([1.57, 0, 0.0], 0.7, dict(down=[0, 2, 3], out_up=[1]))]
    attack = [get_quadruped_pose([1.57, 0., -0.4], 0.7, dict(out=[0, 1, 2, 3]))]
    balance_front = [get_quadruped_pose([1.57, 0.0, 1.57], 0.7, dict(up=[0, 1, 2, 3]))]
    balance_back = [get_quadruped_pose([1.57, 0.0, -1.57], 0.7, dict(up=[0, 1, 2, 3]))]
    balance_diag = [get_quadruped_pose([1.57, 0, 0.0], 0.7, dict(down=[0, 2], out_up=[1,3]))]

    goals = np.stack([ lie_legs_together, lie_rotated, lie_two_legs_up,
                      lie_side, lie_side_back, stand, stand_rotated,
                      stand_leg_up, attack, balance_front, balance_back,  balance_diag])

  return goals


def get_quadruped_pose(global_rot, global_pos=0.5, legs={}, legs_rot=[0, 0, 0, 0]):
  """

  :param angles: along height, along depth, along left-right
  :param height:
  :param legs:
  :return:
  """
  if not isinstance(global_pos, list):
    global_pos = [0, 0, global_pos]
  pose = np.zeros([23])
  pose[0:3] = global_pos
  pose[3:7] = (Rotation.from_euler('XYZ', global_rot).as_quat())

  pose[[7, 11, 15, 19]] = legs_rot
  for k, v in legs.items():
    for leg in v:
      if k == 'out':
        pose[[8 + leg * 4]] = 0.5  # pitch
        pose[[9 + leg * 4]] = -1.0  # knee
        pose[[10 + leg * 4]] = 0.5  # ankle
      if k == 'inward':
        pose[[8 + leg * 4]] = -0.35  # pitch
        pose[[9 + leg * 4]] = 0.9  # knee
        pose[[10 + leg * 4]] = -0.5  # ankle
      elif k == 'down':
        pose[[8 + leg * 4]] = 1.0  # pitch
        pose[[9 + leg * 4]] = -0.75  # knee
        pose[[10 + leg * 4]] = -0.3  # ankle
      elif k == 'out_up':
        pose[[8 + leg * 4]] = -0.2  # pitch
        pose[[9 + leg * 4]] = -0.8  # knee
        pose[[10 + leg * 4]] = 1.  # ankle
      elif k == 'up':
        pose[[8 + leg * 4]] = -0.35  # pitch
        pose[[9 + leg * 4]] = -0.2  # knee
        pose[[10 + leg * 4]] = 0.6  # ankle

  return pose
