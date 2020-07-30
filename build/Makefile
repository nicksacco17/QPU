NVCC := nvcc
EXEC := qpu

all:$(EXEC)

qpu:
	$(NVCC) -o qpu qpu_main.cu ../src/*.cpp -lcusolver -lcublas -DUSE_GPU
clean:
	rm -f *.o qpu
