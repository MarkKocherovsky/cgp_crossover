#!/bin/bash
#mark kocherovsky
#feb 2023
#launch lgp flattened one point crossover

g=10000 #generations
r=64 #rules
d=4 #destinations
p=40 #parents
c=40 #children
ft=1 #fitness function
pm=0.025 #mutation probability
px=0.5 #xover probability
for f in {1..6}
do
	for t in {1..50}
	do
		sbatch lgp_fx.sb $t $g $r $d $p $c $f $ft $pm $px
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
