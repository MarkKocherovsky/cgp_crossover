#!/bin/bash
#mark kocherovsky
#April 2024
#launch cgp 1 point crossover variable length

g=10000
n=64
p=40
c=40
points_simple=20
points_complex=40
dims=1
pm=0.025
pc=0.5
xov=2
for f in {0..6}
do
	for t in {1..50}
	do
		sbatch cgp_vlen.sb $t $g $n $p $c $f $dims $points_simple $pm $pc $xov &
	done
done
for f in {7..10}
do
	for t in {1..50}
	do
		sbatch cgp_vlen.sb $t $g $n $p $c $f $dims $points_complex $pm $pc $xov &
	done
done
