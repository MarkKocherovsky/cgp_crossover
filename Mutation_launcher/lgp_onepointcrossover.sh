#!/bin/bash
#launch cgp base

for f in {0..6}
do
	for t in {1..50}
	do
		sbatch lgp_onepointcrossover.sb $t $f &
	done
done