#!/bin/bash

#Mark Kocherovsky
f=0 #function, [0,8]
t=1 #trials
g=100 #gens
n=24 #nodes
c=4 #children

for t in {1..1}
do
	for f in {0..8}
	do
		sbatch run_cgp_base.sb $f $t $g $n $c &
		#python3 baseline_1_5_rule.py $i $gens
	done
done
