#!/bin/bash
#mark kocherovsky
#dec 2023
#launch lgp one point crossover

g=10000 #generations
r=64 #rules
d=4 #destinations
p=40 #parents
c=40 #children
ft=1 #fitness function
pm=0.025 #mutation probability
px=0.5 #xover probability
rn=""
echo $rn
fl=0 #Fixed length - 1 = True
echo $fl
for f in {2..2}
do
	for t in {1..50}
	do
		sbatch lgp_1x.sb $t $g $r $d $p $c $f $ft $pm $px $rn $fl
		#python3 ../src/lgp.py $t $g $r $d $p $c $f &
	done
done
