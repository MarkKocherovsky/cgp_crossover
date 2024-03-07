#!/bin/bash
#mark kocherovsky
#Feb 2024
#launch cgp Node 1 point crossover

g=10000
n=64
p=40
c=40
ft=1
pm=0.025
pc=0.5
for f in {6..6}
do
	for t in {28..28}
	do
		sbatch cgp_nx.sb $t $g $n $p $c $f $ft $pm $pc &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
