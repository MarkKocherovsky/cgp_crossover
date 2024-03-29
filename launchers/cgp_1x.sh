#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp 1 point crossover

g=10000
n=64
p=40
c=40
ft=1
pm=0.025
pc=0.5
for f in {0..6}  #cgp_1x.py 1 100 64 40 40 0 1 0.025 0.5
do
	for t in {1..50}
	do
		sbatch cgp_1x.sb $t $g $n $p $c $f $ft $pm $pc &
		#python ../src/cgp.py $t $g $n $c $f &
	done
done
