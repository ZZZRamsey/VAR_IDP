#!/bin/bash
# Run this script on Machine 1 (172.16.46.140)
export NODE_RANK=0
bash $(dirname "$0")/train_mul_node.sh
