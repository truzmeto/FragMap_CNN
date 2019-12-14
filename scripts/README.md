# UNIT testing on Amarel

## SSH forwarding and running jupyter

### STEP 1: Log into a GPU node and run Jupyter on it.

` cd scripts `

` mkdir -p jupyter_errs jupyter_outs `

Here `-p` will ignore making the directory if it already exists.

` sbatch scripts/start_jupyter.sh `

This script will run jupyter interactively on a GPU node, based on the header. Currently the header is set to --partition=main and --constraint=pascal

These options alone should ensure you don't have a long queue time.

The script also assigns the jupyter to port # 8889, will need this in STEP 2.

### STEP 2: Tunneling login node jupyter port # to local port #.

Check what node your jupyter is running on.

DO: `squeue -u sb1638 ` but replace sb1638 with your NetID.

Look at the NODELIST column, it will say something like `pascal010`

This is the node your jupyter notebook or jupyter lab is running in.

Log into amarel as follows:

`ssh -L 9999:pascal010:8889 sb1638@amarel.hpc.rutgers.edu` but replace pascal010 with the GPU name you got from NODELIST and sb1638 with your NetID.

This will send the jupyter port 8889 in node pascal010 to your localhost port 9999.

### STEP 3: Launch the interactive Jupyter session.

In your web browswer type into the url bar:

`localhost:9999`

This will open up your session. And you're in.

To check what you're running on type `!nvidia-smi` into a cell in your jupyter notebook.

Now you can unit test on a GPU on amarel.


#### Troubleshooting Notes on Amarel

  In SBATCH header: To figure out what `partition`, `gres`, and `constraint=<feature>`to potentially use:

`sinfo -o "%20P %20N  %10c  %10m  %25f  %10G %20t" | grep 'titan\|pascal'`
    
this shows what to fill into the SBATCH header parameters... 

PARTITION            NODELIST              CPUS        MEMORY      AVAIL_FEATURES             GRES       STATE


main*                gpu007                24          190000      edr,titan                  gpu:8      mix                  
main*                gpu[008-010]          24          190000      edr,titan,oarc             gpu:8      mix                  
main*                pascal[001-002,008-0  28+         128000+     edr,pascal                 gpu:2      mix                  
main*                pascal[003-007]       28+         128000+     edr,pascal,oarc            gpu:2      mix                  
main*                pascal010             32          192000      edr,pascal,oarc            gpu:2      alloc                
gpu                  gpu007                24          190000      edr,titan                  gpu:8      mix                  
gpu                  gpu[008-010]          24          190000      edr,titan,oarc             gpu:8      mix                  
gpu                  pascal[001-002,008-0  28+         128000+     edr,pascal                 gpu:2      mix                  
gpu                  pascal[003-007]       28+         128000+     edr,pascal,oarc            gpu:2      mix                  
gpu                  pascal010             32          192000      edr,pascal,oarc            gpu:2      alloc  	

