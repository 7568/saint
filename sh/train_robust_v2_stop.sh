#!/bin/bash
while IFS= read -r line;
do
  kill -9 $line
done < pid/train_robust_v2.pid