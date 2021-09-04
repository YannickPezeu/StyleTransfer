#!/bin/bash


FILES="origins/*"
for f in $FILES
do
    python main.py $f
    sleep 1
done


