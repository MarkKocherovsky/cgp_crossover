#!/bin/bash
#mark kocherovsky
#dec 2023
#launch cgp 1 point crossover

g=1000
n=5
p=40
c=40
dims=1
pm=0.025
pc=0.5
points_simple=20
points_complex=40
for f in {0..6}
do
        for t in {1..50}
        do
                sbatch cgp_real.sb $t $g $n $p $c $f $dims $points_simple $pm $pc &
                #python3 ../src/cgp.py $t $g $n $c $f &
        done
done
for f in {7..10}
do
        for t in {1..50}
        do
                sbatch cgp_real.sb $t $g $n $p $c $f $dims $points_complex $pm $pc &
                #python3 ../src/cgp.py $t $g $n $c $f &
        done
done

