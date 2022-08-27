#!/bin/bash
cp default.py answer/lexsub.py
cp default.ipynb answer/lexsub.ipynb
python3 zipout.py
python3 check.py