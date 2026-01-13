#!/usr/bin/env bash
if ! screen -list | grep -q "fahrplan"; then
    cd ~/git/github.com/greflm13/fahrplan || exit
    screen -S fahrplan -d -m python api.py
fi