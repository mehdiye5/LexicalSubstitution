#!/bin/bash
python retrofit.py
python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude