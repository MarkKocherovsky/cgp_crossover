#!/bin/bash
#mark kocherovsky
#Jan 2024
#launch cgp(40+40)

g=10000
n=64
p=16
c=4 #4 children for each parent
ft=1
for f in {1..1}
do
	for t in {51..51}
	do
		sbatch cgp_40.sb $t $g $n $p $c $f $ft &
		#python3 ../src/cgp.py $t $g $n $c $f &
	done
done
