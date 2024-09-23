GPU = 1
CPU = 2
T = 600
DATA = "popfva_samples"
EPOCHS = 100
N_FEAT = 256

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "source /home/evkhomutov/kenenbek/mac3/pypypy/bin/activate" >> run.sh;
	echo "srun python run.py --data=$(DATA) --epochs=$(EPOCHS) --features=$(N_FEAT)" >> run.sh;
	sbatch -A proj_1591 --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


hse-run-test:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "source pypypy/bin/activate" >> run.sh;
	echo "srun python brute_force_run.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh
