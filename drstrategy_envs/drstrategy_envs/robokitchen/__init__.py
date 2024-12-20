#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import drstrategy_envs.robokitchen.franka

from drstrategy_envs.robokitchen.utils.configurable import global_config

from gym.envs.registration import register

from .kitchen_envs import (
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
)


register(
    id='kitchen-lexa-v0',
    entry_point='drstrategy_envs.robokitchen:KitchenMicrowaveKettleLightTopLeftBurnerV0',
    max_episode_steps=280,
    )
