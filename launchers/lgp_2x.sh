#!/bin/bash
#mark kocherovsky
#Jan 2024
#launch lgp 2point crossover

g=10000 #generations
r=64 #rules
d=4 #destinations
p=40 #parents
c=40 #children
ft=1
pm=0.025
pc=0.5
for f in {0..6}
do
	for t in {1..50}
	do
		sbatch lgp_2x.sb $t $g $r $d $p $c $f $ft $pm $pc
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
