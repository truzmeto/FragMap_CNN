#!/bin/bash

#### A simple script to change outdated PDB column values.
###########################################################
##      CD stands for delta carbon, but needs to be numbered
##          CD -> CD2
##      HSD is histadine but with different charges
##          HSD -> HIS
##      OT1, OT2 are oxygens, but OXT is the standard naming convention.

## To Run: cd /data
## cp *.pdb into orig_pdbs ### to save original unmodified .pdb files
## DO: bash ../util/updatePDBnames.sh 

cd ../data/orig_pdbs

echo 'Running in' $(pwd)

mkdir -p tmp


########################
##CHECK OUTPUTS
########################

## in /data DO:
## grep 'CD  ILE' *
## grep 'HSD' *
## grep 'OT1' *
## grep 'OT2' *


for i in $(ls *.pdb); 

do 

less $i | sed 's/CD\ \ ILE/CD1 ILE/g' | sed 's/HSD\ /HIS /g' | sed 's/OT1/OXT/g' | sed 's/OT2/OXT/g' > tmp/$i;

mv tmp/$i ~/FragMap_CNN/data;

echo finished $i

done;


echo DONE
