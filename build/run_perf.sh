make
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 1
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 2
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 4
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 8
