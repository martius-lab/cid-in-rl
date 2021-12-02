#!/bin/sh

set -v

find experiments/2-prioritization -iname "*.gin" -exec python -m cid.train {} train.n_iterations=201 \;
find experiments/3-exploration -iname "*.gin" -exec python -m cid.train {} train.n_iterations=201 \;
find experiments/4-other -iname "*.gin" -exec python -m cid.train {} train.n_iterations=201 \;