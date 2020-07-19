#!/bin/bash
#SBATCH -J qpu.%u
#SBATCH -o qpu.%u.o%j
#SBATCH -e qpu.%u.e%j
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -t 04:00:00 # 1 hour

# q == number of qubits in system (2 --> 20)
for ((q=2; q<=15; q++))
do
	nvidia-smi
	./qpu $q
	sleep 1
done
