make
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 1 > grep -v ">>>"
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 2
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 4
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 8
perf stat -B -e cache-references,cache-misses ./Parellel_OpFlow_main -n 16
# ./Parellel_OpFlow_main -n 1
# ./Parellel_OpFlow_main -n 2
# ./Parellel_OpFlow_main -n 4
# ./Parellel_OpFlow_main -n 8
