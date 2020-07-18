#!/bin/bash
#SBATCH -J qpu.%u
#SBATCH -o qpu.%u.o%j
#SBATCH -e qpu.%u.e%j
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -t 04:00:00 # 1 hour

for ((q=2; q<=30; q++))
do
	nvidia-smi
	./qpu $((2**q))
	sleep 1
done

#nvidia-smi
#./qpu

#for i in 1 2
#do
#	for j in 2048 4096 8192 16384
#	do
#		echo "----------ITERATION $i ---------- DIMENSION ($j x $j) ----------"
#		nvidia-smi
#		./gauss $j
#		sleep 10
#	done
#done

#nvidia-smi
#./gauss 32
