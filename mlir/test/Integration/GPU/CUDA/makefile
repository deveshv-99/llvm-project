
all: clean shared run

shared:
	@ clang++ -c ./shared_stuff/shared.cpp -g -o ./shared_stuff/shared.o
	@ clang++ -shared ./shared_stuff/shared.o -o ./shared_stuff/shared.so

test:
	./run_test.sh test-shared.mlir

run:
	./run_test.sh join_v1.mlir

clean:
	@ rm -f ./shared_stuff/shared.o
	@ rm -f ./shared_stuff/shared.so