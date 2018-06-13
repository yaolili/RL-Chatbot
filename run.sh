#!/usr/bin/env bash
# AUTHOR:   yaoll
# FILE:     run.sh
# ROLE:     training, testing, simulation (detailed argument parameters are in config.py) 
# CREATED:  2018-06-12 17:15:35
# MODIFIED: 2018-06-12 17:15:41


# training
python python/train.py -m rlcw

# testing
python python/test.py -m rlcw

# simulation
python python/simulate.py -m rlcw
