NVCC := nvcc
EXEC := qpu

all:$(EXEC)

qpu:
	$(NVCC) -o qpu qpu_main.cu src/Matrix.cpp src/Utility.cpp -lcusolver
clean:
	rm -f *.o qpu
