# feedforward 
sbatch run_train.sh -c configs/ff.json 
sbatch run_train.sh -c configs/ff.json -E


sbatch run_train.sh -c configs/fno_deq.json 
sbatch run_train.sh -c configs/fno_deq.json -E


sbatch run_train.sh -c configs/ifno.json 
sbatch run_train.sh -c configs/ifno.json -E


sbatch run_train.sh -c configs/thermino.json 
sbatch run_train.sh -c configs/thermino.json -E

sbatch run_train.sh -c configs/thermino_notherm.json 
sbatch run_train.sh -c configs/thermino_notherm.json -E

sbatch run_train.sh -c configs/thermino_pre.json 
sbatch run_train.sh -c configs/thermino_post.json 

# sbatch run_train.sh -c configs/ff.json 
# sbatch run_train.sh -c configs/ff.json -E

# # feedforward 
# sbatch run_train.sh -c configs/fno_FF_Cflat.json 
# # sbatch run_train.sh -c configs/fno_FF_Cflat.json -E

# # DEQ
# sbatch run_train.sh -c configs/fno_deq.json 
# sbatch run_train.sh -c configs/fno_deq.json -E

# sbatch run_train.sh -c configs/fno_deq_strainonly.json
# sbatch run_train.sh -c configs/fno_deq_stressonly.json
# sbatch run_train.sh -c configs/fno_deq_energyonly.json

# sbatch run_train.sh -c configs/fno_deq_pre.json 
# # sbatch run_train.sh -c configs/fno_deq_pre.json -E

# sbatch run_train.sh -c configs/fno_deq_post.json 
# # sbatch run_train.sh -c configs/fno_deq_post.json -E

# sbatch run_train.sh -c configs/fno_deq_nopolar.json 
# # sbatch run_train.sh -c configs/fno_deq_nopolar.json -E

# sbatch run_train.sh -c configs/fno_deq_noenergy.json 

# sbatch run_train.sh -c configs/fno_deq_nothermo.json 
# sbatch run_train.sh -c configs/fno_deq_nothermo.json -E

# sbatch run_train.sh -c configs/fno_deq_hybrid.json 
# sbatch run_train.sh -c configs/fno_deq_hybrid.json -E
# # Hybrid DEQ
# sbatch run_train.sh -c configs/fno_deq_hybrid.json 
# sbatch run_train.sh -c configs/fno_deq_hybrid.json -E

# # DEQ, no thermo features
# sbatch run_train.sh -c configs/fno_deq_nothermo.json 
# # sbatch run_train.sh -c configs/fno_deq_nothermo.json -E

# # DEQ, no energy features
# sbatch run_train.sh -c configs/fno_deq_noenergy.json 
# # sbatch run_train.sh -c configs/fno_deq_noenergy.json -E

# # DEQ, no polar features
# sbatch run_train.sh -c configs/fno_deq_nopolar.json 
# sbatch run_train.sh -c configs/fno_deq_nopolar.json -E

# # run big model with everything
# sbatch run_train.sh -c configs/maximalist.json 
# sbatch run_train.sh -c configs/maximalist.json -E