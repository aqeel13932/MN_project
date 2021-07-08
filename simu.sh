#22/04/2020 Added simulations for jetlag, first day disturbtion, random start of day.(1000 episode each)
#12/06/2020 Added simulations for PRC 
#06/10/2020 Added simulations for 50 days and 1 time step manipluation for 3rd day and 7th day.
#08/10/2020 added simulations for 50 days and 1 time steo manipluation for 3rd day and 7th day for model 67.
#09/10/2020 aaded simulations for jetlage (moving west,moving east) every time step for (3rd,7th) day for both (64,67) models.
: <<'END'
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 30 --num_eps 1000 --nofood  >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 39 --num_eps 1000 --nofood --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 41 --num_eps 1000 --nofood --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 45 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 47 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 51 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 53 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 2 --num_eps 1000 --rwrdschem 0 1 -1.5 --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 6 --num_eps 1000 --rwrdschem 0 1 -1.5 --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 28 --num_eps 1000 --clue >logs.txt &
for i in `seq 57 63`;
do 
	echo $i
	nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m $i --num_eps 1000 --clue >logs.txt &
done

for i in `seq 64 70`;
do 
	echo $i
	nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m $i --max_timesteps 160 --num_eps 1000 --clue >logs.txt &
done
for i in `seq 64 70`;
do 
	echo $i
	nohup srun --partition=main --time=1- --mem=6000 python manipulated_mn_record.py --train_m $i --max_timesteps 160 --manipulation 10000000 --num_eps 100 --clue >logs_10.txt &
	nohup srun --partition=main --time=1- --mem=6000 python manipulated_mn_record.py --train_m $i --max_timesteps 160 --manipulation 10100000 --num_eps 100 --clue >logs_1010.txt &
	nohup srun --partition=main --time=1- --mem=6000 python manipulated_mn_record.py --train_m $i --max_timesteps 160 --manipulation 11111111 --num_eps 100 --clue >logs_1.txt &
	nohup srun --partition=main --time=1- --mem=6000 python manipulated_mn_record.py --train_m $i --max_timesteps 160 --manipulation 00000000 --num_eps 100 --clue >logs_0.txt &
done

for i in `seq 64 70`;
do 
	echo $i
	#normal
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1100110011001100 --num_eps 1000 --clue >logs_10.txt &
	#disturb first day, disturb the morning
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1110110011001100 --num_eps 1000 --clue >logs_10.txt &
	#disturb first day, disturb the night
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1000110011001100 --num_eps 1000 --clue >logs_10.txt &
	#add jet lag, longer night for day 2
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1100111001100110 --num_eps 1000 --clue >logs_10.txt &
	#add jet lag, longer morning for day 2
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1100110001100110 --num_eps 1000 --clue >logs_10.txt &
	# Random configuration (start at different time of morning or night)
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 1001100110011001 --num_eps 1000 --clue >logs_10.txt &
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 0011001100110011 --num_eps 1000 --clue >logs_10.txt &
	nohup srun --partition=amd --time=1- --mem=5000 python counter_timer_mn_record.py --train_m $i --max_timesteps 160 --manipulation 0110011001100110 --num_eps 1000 --clue >logs_10.txt &
done


for i in `seq 64 70`;
do 
	for j in `seq 20 21`;
		do
			echo $i $j
			nohup srun --partition=amd --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m $i --episode_length 320 --Scenario $j --num_eps 1000 --clue >logs/"${i}_${j}.out" &
		done
done
for i in `seq 64 70`;
do 
	for j in `seq 0 21`;
		do
			echo $i $j
			nohup srun --partition=amd --time=3- --mem=20000 python PRC_dynamic_timer_mn_record.py --train_m $i --episode_length 320 --Scenario $j --num_eps 100 --clue >logs/"${i}_${j}.out" &
		done
done

nohup srun --partition=amd --time=3- --mem=30000 python dynamic_timer_mn_record.py --train_m 64 --episode_length 2000 --Scenario 22 --num_eps 1000 --clue >logs/64_22.out &
nohup srun --partition=amd --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m 64 --episode_length 320 --Scenario 23 --num_eps 1000 --clue >logs/64_23.out &
: <<'END'
# Test the pretrubation:
for j in `seq 23 102`;
	do
		echo $j
		nohup srun --partition=amd --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m 64 --episode_length 320 --Scenario $j --num_eps 1000 --clue >logs/64_"${j}.out" &
	done
# some simulations got stuck, redoing them with different cluster option.
exp=(54 57 58 59 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102)
for j in ${exp[@]};
	do
		echo $j
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m 64 --episode_length 320 --Scenario $j --num_eps 1000 --clue >logs/64_"${j}.out" &
	done
nohup srun --partition=main --time=3- --mem=30000 python dynamic_timer_mn_record.py --train_m 67 --episode_length 2000 --Scenario 22 --num_eps 1000 --clue >logs/67_22.out &
for j in `seq 23 102`;
	do
		echo $j
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m 67 --episode_length 320 --Scenario $j --num_eps 1000 --clue >logs/67_"${j}.out" &
	done

mod=(64 67)
for m in ${mod[@]};
do
	for s in `seq 104 104`;
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=30000 python dynamic_timer_mn_record.py --train_m $m --episode_length 2000 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done
done


mod=(64 67)
for m in ${mod[@]};
do
	for s in `seq 105 264`;
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record.py --train_m $m --episode_length 400 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done
done
END

mod=(78 81)
exp=(22 103 104)
for m in ${mod[@]};
do
	for s in `seq 0 21`;
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record_minimized.py --train_m $m --episode_length 320 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done

	for s in `seq 23 102`;
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record_minimized.py --train_m $m --episode_length 320 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done

	for s in ${exp[@]};
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=15000 python dynamic_timer_mn_record_minimized.py --train_m $m --episode_length 2000 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done

	for s in `seq 105 264`;
	do
		echo $m $s
		nohup srun --partition=main --time=3- --mem=5000 python dynamic_timer_mn_record_minimized.py --train_m $m --episode_length 400 --Scenario $s --num_eps 1000 --clue >logs/"${m}_${s}.out" &
	done
done
