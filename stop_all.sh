#!/bin/bash
while IFS= read -r line;
do
  kill -9 $line
done < pid/train_v2.pid