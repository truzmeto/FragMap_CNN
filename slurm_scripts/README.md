
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

