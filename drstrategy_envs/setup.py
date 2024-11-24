from setuptools import setup, find_packages

setup(
    name='drstrategy_envs',
    version='0.0.1',
    keywords='environment, agent, rl, openaigym, openai-gym, gym',
    packages=find_packages(),
    install_requires=[
        'gym[accept-rom-license, atari]>=0.23.1',
        'numpy>=1.19.5',
        'pyglet==1.5.24',
        'matplotlib>=3.4.2',
        'imagehash==4.3.1',
        'numpy-quaternion==2022.4.3',
        'termcolor'
    ],
    extras_require={'gui': ['pygame', 'Pillow']},
    entry_points={'console_scripts': []},
    # Include textures and meshes in the package
    include_package_data=True
)
