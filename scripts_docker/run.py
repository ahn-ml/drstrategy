#!/usr/bin/python3

import os
import argparse
import random
from datetime import datetime

username = os.path.expanduser('~').split('/')[-1]
to_path = "root"
image = username+"/drstrategy"

random.seed(datetime.now().timestamp())

parser = argparse.ArgumentParser(
    prog="run_docker.py", description="Runner", epilog="Help"
)

parser.add_argument("--device", nargs="+", type=int, default=[0], required=False)
parser.add_argument("--script", type=str, default="bash", required=False)
parser.add_argument("--bindport", type=str, default="", required=False)


args = parser.parse_args()

devices = [str(d) for d in args.device]
key = ","
device = f"'\"device={key.join(devices)}\"'"
script = args.script
text = f"WANDB_API_KEY=$WANDB_API_KEY && docker run -it {args.bindport} --rm --gpus {device} --user=$(id -u):$(id -g) -v $(pwd):/{to_path} --env WANDB_API_KEY=$WANDB_API_KEY --env HostName=$(hostname) --env ServerNum={''.join(devices)} {image} {script}"

print(text)
os.system(text)
