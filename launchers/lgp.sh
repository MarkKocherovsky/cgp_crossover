#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp base

g=10000 #generations
r=64 #rules
d=4 #destinations
p=40 #parents
c=40 #children
ft=1
pm=0.025
px=0.5

for f in {2..2}
do
	for t in {15 28}
	do
		sbatch lgp.sb $t $g $r $d $p $c $f $ft $pm $px
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
