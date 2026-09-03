#!/bin/bash

 yum update -y
 amazon-linux-extras install docker
 systemctl start docer
 systemctl enable docker

 wget https://raw.githubusercontent.com/LLNL/RAJA/task/tut-reorg-aws/exercises/Dockerfile
 wget https://raw.githubusercontent.com/LLNL/RAJA/task/tut-reorg-aws/exercises/supervisord.conf

 env DOCKER_BUILDKIT=1 docker build . -t raja-aws-tut
 docker run --init --gpus all -p 3000:3000 raja-aws-tut