#!/bin/sh

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat
cd ${netmat2surf_path}
sbatch ./tools/SiT_LN/submit_krakenSiT_train_surf2surfrecon.sh # submits the first one which is chnl==0
echo "Just submitted channel 0"
# where the config files are
yaml_loc="${netmat2surf_path}/config/SiT_LN"

for ii in {0..13}; #0..13 cause it adds 1 and so at 13-->14
do
    sleep 2m 30s
    cd ${yaml_loc}

    initial=$ii # then changes channel==0 -> channel==1 and loops from 1-14
    next=$(($ii + 1))
    echo ${initial} ${next}

    sed -i "s/\bspecific_channel: ${initial}\b/specific_channel: ${next}/g" "${yaml_loc}/hparams_krakenSiTLN_surf2surfrecon.yml"
    sed -i "s/\bspecific_channel_end: ${initial}\b/specific_channel_end: ${next}/g" "${yaml_loc}/hparams_krakenSiTLN_surf2surfrecon.yml"

    # wait a bit before sening next so that the channel change applies
    # and is locked in the cache, i.e. making sure all have different log files. 
    sleep 2m 30s 

    cd ${netmat2surf_path}/tools/SiT_LN/
    sbatch submit_krakenSiT_train_surf2surfrecon.sh
    echo "Just submitted channel ${next}"

done

echo DONE