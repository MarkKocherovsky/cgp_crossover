#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp base

g=10000 #generations
r=64 #rules
d=12 #destinations
p=40 #parents
c=40 #children

for f in {0..6}
do
	for t in {2..50}
	do
		sbatch lgp_1x.sb $t $g $r $d $p $c $f
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
