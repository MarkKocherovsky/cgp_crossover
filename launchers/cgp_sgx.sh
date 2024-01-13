#!/bin/bash
#mark kocherovsky
#jan 2024
#launch cgp subgraph crossover

g=100
n=64
p=40
c=40

for f in {0..6}
do
	for t in {1..1}
	do
		sbatch cgp_sgx.sb $t $g $n $p $c $f &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
