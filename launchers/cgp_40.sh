#!/bin/bash
#mark kocherovsky
#Jan 2024
#launch cgp(40+40)

g=10000
n=64
p=40
c=40

for f in {0..6}
do
	for t in {2..50}
	do
		sbatch cgp_40.sb $t $g $n $p $c $f &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
