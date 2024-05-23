# feedforward 
sbatch run_train.sh -c configs/fno_FF.json 
sbatch run_train.sh -c configs/fno_FF.json -E
# DEQ
sbatch run_train.sh -c configs/fno_deq.json 
sbatch run_train.sh -c configs/fno_deq.json -E
# Hybrid DEQ
sbatch run_train.sh -c configs/fno_deq_hybrid.json 
sbatch run_train.sh -c configs/fno_deq_hybrid.json -E

# DEQ, no thermo features
sbatch run_train.sh -c configs/fno_deq_nothermo.json 
sbatch run_train.sh -c configs/fno_deq_nothermo.json -E

# DEQ, no energy features
sbatch run_train.sh -c configs/fno_deq_noenergy.json 
sbatch run_train.sh -c configs/fno_deq_noenergy.json -E

# DEQ, no polar features
sbatch run_train.sh -c configs/fno_deq_nopolar.json 
sbatch run_train.sh -c configs/fno_deq_nopolar.json -E


sbatch run_train.sh -c configs/maximalist.json 