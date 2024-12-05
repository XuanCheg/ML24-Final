dset_name=$1
results_root=results
exp_id=exp

data_root=data

#### training
bsz=4
truncate=6
model_name_or_path=./Llama-2-7b-hf
n_epoch=100
lr=1e-4
wd=1e-4

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python main.py \
--dset_name ${dset_name} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--data_root ${data_root} \
--truncate ${truncate} \
--model_name_or_path ${model_name_or_path} \
--n_epoch ${n_epoch} \
--lr ${lr} \
--wd ${wd} \
--num_workers 0 \
${@:2}