#!/bin/bash
#launch cgp base

for f in {0..6}
do
	for t in {1..50}
	do
		sbatch AdaptiveMutation_twopointcrossover.sb $t $f &
	done
done