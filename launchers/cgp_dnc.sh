#!/bin/bash
#mark kocherovsky
#August 2024
#launch cgp deep neural crossover

g=10000
n=64
p=40
pm=0.025
pc=0.5
xov=1
lr=1e-4
dims=1
points_simple=20
points_complex=40

for f in {0..6}
do
	for t in {1..50}
	do
		sbatch cgp_dnc.sb $t $g $n $p $f $dims $points_simple $pm $pc $xov $lr &
	done
done
for f in {7..10}
do
	for t in {1..50}
	do
		sbatch cgp_dnc.sb $t $g $n $p $f $dims $points_complex $pm $pc $xov $lr &
	done
done
