#!/bin/bash
#mark kocherovsky
#April 2024
#launch cgp 1 point crossover variable length

g=10000
n=64
p=40
c=40
ft=1
pm=0.025
pc=0.5
for f in {1..1}
do
	for t in {51..51}
	do
		sbatch cgp_vlen.sb $t $g $n $p $c $f $ft $pm $pc &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
