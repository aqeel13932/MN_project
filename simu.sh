#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 30 --num_eps 1000 --nofood  >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 39 --num_eps 1000 --nofood --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 41 --num_eps 1000 --nofood --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 45 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 47 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 51 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 --clue >logs.txt &
#nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 53 --num_eps 1000 --nofood --rwrdschem 0 1000 -100 --clue >logs.txt &
nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 2 --num_eps 1000 --rwrdschem 0 1 -1.5 --clue >logs.txt &
nohup srun --partition=main --time=1- --mem=6000 python mn_record.py --train_m 6 --num_eps 1000 --rwrdschem 0 1 -1.5 --clue >logs.txt &
