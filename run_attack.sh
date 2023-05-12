#!bin/bash
for attack in 0.05 0.1 0.15 0.2
do
	echo "The current attack rate is:$attack"
	for i in 1 2 3 4 5 6 7 8 9 10
	do
		echo "$i runs $arrack"
		nohup python -u train_GRACE.py --attack_rate $attack > ./run_rate_${attack}_${i}_runs.log 2>&1 &
		wait
	done
done
