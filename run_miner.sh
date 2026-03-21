#!/bin/bash
cd /root/vidaio-subnet
export PYTHONPATH=/root/vidaio-subnet
exec venv/bin/python neurons/miner.py   --wallet.name LukeTao   --wallet.hotkey default   --subtensor.network finney   --netuid 85   --axon.port 40000   --logging.debug
