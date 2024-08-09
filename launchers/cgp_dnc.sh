#!/bin/bash
#mark kocherovsky
#August 2024
#launch cgp deep neural crossover

g=10000
n=64
p=80
ft=1
pm=0.025
pc=0.5
for f in {1..1}
do
	for t in {51..51}
	do
		sbatch cgp_dnc.sb $t $g $n $p $f $ft $pm $pc &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
