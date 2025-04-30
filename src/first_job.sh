#!/bin/bash --login
echo $pwd
rm /tmp/ckpt/*
rm /mnt/gs21/scratch/kocherov/ckpt/*
rm ../output/logs/first_submission_flag 
rm checkpoint.json 
sbatch schedule_jobs.sb
