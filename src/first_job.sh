#!/bin/bash --login
echo pwd
rm ../output/ckpt/*
rm ../output/logs/first_submission_flag 
rm checkpoint.json 
sbatch schedule_jobs.sb
