#!/bin/bash

mkdir -p tmp
cp orig_pdbs/*.pdb ./
for i in $(ls *.pdb); do sed 's/CD\ \ ILE/CD1 ILE/g' $i > tmp/$i; mv tmp/* ./;  done
for i in $(ls *.pdb); do sed 's/HSD\ /HIS /g' $i > tmp/$i; mv tmp/* ./ ;  done
for i in $(ls *.pdb); do sed 's/OT1\ LEU/OXT LEU/g' $i > tmp/$i; mv tmp/* ./;  done
for i in $(ls *.pdb); do sed 's/OT2\ LEU/OXT LEU/g' $i > tmp/$i; mv tmp/* ./;  done

