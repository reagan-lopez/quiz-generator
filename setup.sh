#!/bin/bash
conda create -p venv -y python=3.11
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
