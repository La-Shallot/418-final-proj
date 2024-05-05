#!/bin/bash
make
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 1 | grep -v -E '.*>>>.*'
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 2 | grep -v -E '.*>>>.*'
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 4 | grep -v -E '.*>>>.*'
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 8 | grep -v -E '.*>>>.*'
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 16 | grep -v -E '.*>>>.*'
# ./Parellel_OpFlow_main -n 1
# ./Parellel_OpFlow_main -n 2
# ./Parellel_OpFlow_main -n 4
# ./Parellel_OpFlow_main -n 8
