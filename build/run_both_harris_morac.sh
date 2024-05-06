echo "Moravac\n"
./Parellel_OpFlow_main -n 1 -k m| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 2 -k m| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 4 -k m| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 8 -k m| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 16 -k m| grep -v -E '.*>>>.*'
echo "Harris\n"
./Parellel_OpFlow_main -n 1 -k h| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 2 -k h| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 4 -k h| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 8 -k h| grep -v -E '.*>>>.*'
./Parellel_OpFlow_main -n 16 -k h| grep -v -E '.*>>>.*'