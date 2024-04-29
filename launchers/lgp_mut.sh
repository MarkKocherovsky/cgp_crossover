#!/bin/bash
#mark kocherovsky
#Mar 2023
#launch lgp(1+4)

g=10000 #generations
r=64 #rules
d=4 #destinations
p=1 #parents
c=4 #children
ft=1
pm=1.0
px=0.5

for f in {0..6}
do
	for t in {1..50}
	do
		sbatch lgp_mut.sb $t $g $r $d $p $c $f $ft $pm $px
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
