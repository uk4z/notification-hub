#!/bin/bash

pip install -r requirements.txt
huggingface-cli login
huggingface-cli whoami
