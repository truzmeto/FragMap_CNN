#!/bin/bash

#### A simple script to change outdated PDB column values.
###########################################################
##      CD stands for delta carbon, but needs to be numbered
##          CD -> CD2
##      HSD is histadine but with different charges
##          HSD -> HIS
##      OT1, OT2 are oxygens, but OXT is the standard naming convention.

mkdir -p tmp
cp orig_pdbs/*.pdb ./
for i in $(ls *.pdb); do sed 's/CD\ \ ILE/CD1 ILE/g' $i > tmp/$i; mv tmp/* ./;  done
for i in $(ls *.pdb); do sed 's/HSD\ /HIS /g' $i > tmp/$i; mv tmp/* ./ ;  done
for i in $(ls *.pdb); do sed 's/OT1\ LEU/OXT LEU/g' $i > tmp/$i; mv tmp/* ./;  done
for i in $(ls *.pdb); do sed 's/OT2\ LEU/OXT LEU/g' $i > tmp/$i; mv tmp/* ./;  done

