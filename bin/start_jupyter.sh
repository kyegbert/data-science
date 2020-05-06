#!/usr/bin/env bash

. /data_science/.venv/bin/activate
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --notebook-dir=notebooks