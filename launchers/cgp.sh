#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp base

g=10000
n=64
c=4

for f in {0..6}
do
	for t in {1..50}
	do
		sbatch cgp.sb $t $g $n $c $f &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
