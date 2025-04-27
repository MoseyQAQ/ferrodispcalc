#!/bin/bash
BINPATH=$1

for dir in */; do 
    cd $dir
    bash test.sh $BINPATH
    cd ..
done