#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp base

g=120 #generations
r=24 #rules
d=4 #destinations
p=40 #parents
c=40 #children

for f in {0..2}
do
	for t in {1..1}
	do
		sbatch lgp.sb $t $g $r $d $p $c $f
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
