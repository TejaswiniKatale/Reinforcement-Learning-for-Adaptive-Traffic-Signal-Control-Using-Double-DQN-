#!/usr/bin/env bash
set -e
netconvert -n nodes.xml -e edges.xml -o intersection.net.xml
echo "Train: python dqn_controller.py --train --episodes 50"
echo "Eval : python dqn_controller.py --eval --model dqn_model.pth --gui"
