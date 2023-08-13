


##################################################################### 
# Parameters!
mainFolder="net_runs"
subFolder="ECO_full_finetune_UCF101_run1"
snap_pref="eco_full _finetune_UCF101"




train_path="list/ucf101_train_split1.txt"
val_path="list/ucf101_val_split1.txt"



n2D_model="nll"
n3D_model="nll"

nECO_model="ECO_Full_rgb_model_Kinetics.pth.tar"
#############################################
#--- training hyperparams ---
dataset_name="ucf101"
netType="ECOfull"
batch_size=16
learning_rate=0.001
num_segments=16
dropout=0
iter_size=4
num_workers=2

##################################################################### 
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training

echo "Current network folder: "
echo ${mainFolder}/${subFolder}


##################################################################### 
# Find the latest checkpoint of network 
checkpointIter="$(ls ${mainFolder}/${subFolder}/*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
##################################################################### 


echo "${checkpointIter}"

##################################################################### 
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 5 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts finetune --no_partialbn  --nesterov "True" --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt    

else
     echo "Training with initialization"

python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path} --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 5 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts finetune --no_partialbn --nesterov "True" --net_model2D ${n2D_model} --net_model3D ${n3D_model} --net_modelECO ${nECO_model} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

fi

##################################################################### 


