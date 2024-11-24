import argparse
from pathlib import Path

import numpy as np

from memory_maze import helpers, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_camera', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode_start', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--num_random_actions', type=int, default=20)
    parser.add_argument('--episode_len', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='/common/home/fd213/datasets/memory-maze-3x3-coord-reset')
    args = parser.parse_args()

    kwargs = {}
    if args.top_camera:
        args.save_dir = f'{args.save_dir}-top-down'
        kwargs['camera_resolution'] = 256

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = tasks.memory_maze_single_room_3x3(global_observables=True, top_camera=args.top_camera, **kwargs)
    num_actions = 6
    maze_size = 3

    seed = args.seed * (args.episode_start + 1)
    np.random.seed(seed)
    seed = np.random.randint(2147483648)
    np.random.seed(seed)

    episode_idx = args.episode_start
    while episode_idx < args.episode_start + args.num_episodes:
        timestep = env.reset()

        action_repeat = 2
        for _ in range(args.num_random_actions // action_repeat):
            action = np.random.randint(1, num_actions)
            for _ in range(action_repeat):
                timestep = env.step(action)
                assert not timestep.last()

        obs_list = [timestep.observation['image']]
        action_list = []
        step_counter = 0

        while True:
            if (step_counter == 0 and np.random.randint(1, 4) != 1) or (step_counter > 0 and action == 1):
                agent_pos = timestep.observation['agent_pos']
                agent_dir = timestep.observation['agent_dir']
                x, y = agent_pos[0], agent_pos[1]
                dx, dy = agent_dir[0], agent_dir[1]
                left_close = x < 1 and dx < 0
                right_close = x >= 2 and dx >= 0
                bottom_close = y < 1 and dy < 0
                top_close = y >= 2 and dy >= 0
                if (left_close and bottom_close) or (left_close and top_close) or (right_close and bottom_close) or (right_close and top_close):
                    action = np.random.randint(2, 4)
                    action_repeat = np.random.randint(8, 13)
                elif left_close and dy < 0:
                    action = 2
                    action_repeat = np.random.randint(5, 8)
                elif left_close and dy >= 0:
                    action = 3
                    action_repeat = np.random.randint(5, 8)
                elif right_close and dy < 0:
                    action = 3
                    action_repeat = np.random.randint(5, 8)
                elif right_close and dy >= 0:
                    action = 2
                    action_repeat = np.random.randint(5, 8)
                elif bottom_close and dx < 0:
                    action = 3
                    action_repeat = np.random.randint(5, 8)
                elif bottom_close and dx >= 0:
                    action = 2
                    action_repeat = np.random.randint(5, 8)
                elif top_close and dx < 0:
                    action = 2
                    action_repeat = np.random.randint(5, 8)
                elif top_close and dx >= 0:
                    action = 3
                    action_repeat = np.random.randint(5, 8)
                else:
                    action = np.random.randint(2, 4)
                    action_repeat = np.random.randint(0, 3)
            else:
                action = 1
                action_repeat = np.random.randint(2, 5)

            for _ in range(action_repeat):
                timestep = env.step(action)
                assert not timestep.last()
                obs_list.append(timestep.observation['image'])
                agent_pos = timestep.observation['agent_pos'] / maze_size * 2 - 1
                agent_dir = timestep.observation['agent_dir']
                map_coord = np.concatenate((agent_pos, agent_dir), axis=-1)
                action_list.append(map_coord)
                step_counter += 1
                if step_counter == args.episode_len:
                    break
                if action == 1:
                    agent_pos = timestep.observation['agent_pos']
                    agent_dir = timestep.observation['agent_dir']
                    x, y = agent_pos[0], agent_pos[1]
                    dx, dy = agent_dir[0], agent_dir[1]
                    left_close = x < 1 and dx < 0
                    right_close = x >= 2 and dx >= 0
                    bottom_close = y < 1 and dy < 0
                    top_close = y >= 2 and dy >= 0
                    if left_close or right_close or bottom_close or top_close:
                        break

            if step_counter == args.episode_len:
                observations = np.stack(obs_list, axis=0)
                actions = np.stack(action_list, axis=0)
                dummy_action = np.zeros_like(actions[:1])
                actions = np.concatenate((dummy_action, actions), axis=0)
                data = {'image': observations, 'action': actions}
                path = save_dir / f'episode-{episode_idx:06}.npz'
                with path.open('wb') as f:
                    np.savez_compressed(f, **data)
                print(f'Episode {episode_idx:06} done.')
                episode_idx += 1
                break


if __name__ == '__main__':
    main()
