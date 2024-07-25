#!/bin/bash

ude_dir="mpUDE-prostate"
while read -u 10 p; do
  fname="./$ude_dir/$p"
  echo "processing: $fname"
  unzip $fname
done 10<mpUDE-prostate.txt
