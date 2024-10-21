# source env/bin/activate

# save images 
python3 percolation.py --size 10 --experiments 1 --prob_start 0.2 --prob_end 0.8 --prob_step 0.05 --path percolation_results/

# statistic only
# python3 percolation.py --size 10 --experiments 100 --prob_start 0.1 --prob_end 0.9 --prob_step 0.01
