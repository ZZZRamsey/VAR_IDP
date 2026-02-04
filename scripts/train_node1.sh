#!/bin/bash
# Run this script on Machine 2 (172.16.46.142)
export NODE_RANK=1
bash $(dirname "$0")/train_mul_node.sh
