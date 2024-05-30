
all: clean shared test

shared:
	@ clang++ -c ./shared_stuff/shared.cpp -g -fPIC -o ./shared_stuff/shared.o
	@ clang++ -shared ./shared_stuff/shared.o -o ./shared_stuff/shared.so

zone:
	./run_test.sh test-zone1.mlir

zone2:
	./run_test.sh test-zone2.mlir

test:
	./run_test_ll.sh test-shared.mlir > test-shared.ll

testf:
	./run_test.sh test-shared.mlir
run:
	./run_test.sh join_v1.mlir

clean:
	@ rm -f ./shared_stuff/shared.o
	@ rm -f ./shared_stuff/shared.so