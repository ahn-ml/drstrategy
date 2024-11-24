#!/bin/sh
# TODO: check if the following cause any circular dependency or any type of problem (most probably no) 
# Build rlx docker image first
# ../rlx/scripts/build_docker.sh
# each rlx builds for each user, so we need to enter the user to "From" in the dockerfile, this is quick bad fix
sed -i "s/USERNAME/$(whoami)/g" Dockerfile
docker build  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f  Dockerfile -t  $USER/drstrategy .
sed -i "s/$(whoami)/USERNAME/g" Dockerfile