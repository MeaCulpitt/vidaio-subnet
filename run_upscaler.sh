#!/bin/bash
cd /root/vidaio-subnet
export PYTHONPATH=/root/vidaio-subnet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv/bin/python services/upscaling/server.py
