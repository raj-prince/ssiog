#!/bin/bash

set -x

rm metrics.csv

echo "Invoking without metrics logging enabled..."
time python3 training.py --prefix ~/local_data/  --epochs 1 --steps 2 --sample-size 1048576 --batch-size 20000 --log-level INFO --background-threads 96 --read-order FullRandom
echo "validate no metrics.csv"
ls metrics.csv


echo "Invoking with metrics logging enabled..."
time python3 training.py --prefix ~/local_data/  --epochs 1 --steps 2 --sample-size 1048576 --batch-size 20000 --log-level INFO --background-threads 96 --read-order FullRandom --log-metrics=true
echo "validate there is a metrics.csv"
ls metrics.csv
