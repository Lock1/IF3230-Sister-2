NVCC_OPTIONS := "-Wall -Wextra"

serial:
	gcc -Wall -Wextra -o bin/serial other/serial.c

parallel:
	nvcc --compiler-options=${NVCC_OPTIONS} src/parallel.cu -o bin/parallel

test-serial: serial
	cat other/testcase/K04-06-TC1 | ./bin/serial > result/K04-06-TC1_serial.txt
	cat other/testcase/K04-06-TC2 | ./bin/serial > result/K04-06-TC2_serial.txt
	cat other/testcase/K04-06-TC3 | ./bin/serial > result/K04-06-TC3_serial.txt
	cat other/testcase/K04-06-TC4 | ./bin/serial > result/K04-06-TC4_serial.txt

test-parallel: parallel
	./bin/parallel other/testcase/K04-06-TC1 > result/K04-06-TC1_parallel.txt
	./bin/parallel other/testcase/K04-06-TC2 > result/K04-06-TC2_parallel.txt
	./bin/parallel other/testcase/K04-06-TC3 > result/K04-06-TC3_parallel.txt
	./bin/parallel other/testcase/K04-06-TC4 > result/K04-06-TC4_parallel.txt

diff-test: test-serial test-parallel
	diff result/K04-06-TC1_serial.txt result/K04-06-TC1_parallel.txt
	diff result/K04-06-TC2_serial.txt result/K04-06-TC2_parallel.txt
	diff result/K04-06-TC3_serial.txt result/K04-06-TC3_parallel.txt
	diff result/K04-06-TC4_serial.txt result/K04-06-TC4_parallel.txt
