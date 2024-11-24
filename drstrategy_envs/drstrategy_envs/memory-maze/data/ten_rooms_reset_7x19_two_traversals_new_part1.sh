export MUJOCO_GL=egl

MUJOCO_EGL_DEVICE_ID=0 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 27250 &
MUJOCO_EGL_DEVICE_ID=1 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 27500 &
MUJOCO_EGL_DEVICE_ID=2 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 27750 &
MUJOCO_EGL_DEVICE_ID=3 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 28000 &
MUJOCO_EGL_DEVICE_ID=4 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 28250 &
wait

MUJOCO_EGL_DEVICE_ID=0 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 28500 &
MUJOCO_EGL_DEVICE_ID=1 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 28750 &
MUJOCO_EGL_DEVICE_ID=2 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 29000 &
MUJOCO_EGL_DEVICE_ID=3 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 29250 &
MUJOCO_EGL_DEVICE_ID=4 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 29500 &
wait

MUJOCO_EGL_DEVICE_ID=0 python ten_rooms_reset_7x19_two_traversals_new.py --episode_start 29750 &
wait
