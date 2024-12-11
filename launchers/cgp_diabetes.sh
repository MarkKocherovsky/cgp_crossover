#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp 1 point crossover

g=3000
n=64
p=40
c=40
pm=0.025
pc=0.5
for t in {1..50}
do
	sbatch cgp_diabetes.sb $t $g $n $p $c $f $pm $pc &
	#python3 ../src/cgp.py $t $g $n $c $f &
done
