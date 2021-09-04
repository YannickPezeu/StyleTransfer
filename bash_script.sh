#!/bin/bash
# This script performs the transformation of all the files in "origins/" with the style Monet1-.jpg

FILES="origins/*"
for f in $FILES
do
    python main.py Monet1-.jpg $f 
    sleep 1
done


