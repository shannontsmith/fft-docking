# Project overview

<p>This was a fun pet project that I worked on briefly during the COVID-19 pandemic that deals with Fourier Transform-based ligand docking. The work for this was based on the following papers, which outlined the initial conception of this method for protein-protein interactions[1] then subsequently expanded for practical applications by changing the representation allowing faster sampling[2]. </p> 

<p>
In its current state, this script takes in a protein PDB and ligand PDB file separately then each PDB is put into a grid based on gaussian smoothing function applied based on atomic radii, then FFTs are computed on this grid. 
Now that both the protein and ligand FFTs have been computed, we don't have to recompute these. Instead, we can now simply transform the ligand FFT according to the rotational setting provided as flags and compute the resulting IFFT. The user can specify the number of output ligand PDB structures desired and they will be output in order based of IFFT score. 
</p>

This script is simply python3-based and does not carry any chemical information in its current state. <br>
Required python packages: <br>
numpy, matplotlib, scipy, mrcfile <br>

# File descriptions
## *fft_opencl.py*
<p>Basic usage: python fft_opencl.py -p 3pbl_pocketonly.pdb -l ETQ.pdb ## Absolute basic usage <br>
Recommended usage: python fft_opencl.py -p 3pbl_pocketonly.pdb -l ETQ.pdb -angs_per_vox 1 -rot_angle 60 -opencl <br>
Flags to play around with: -rot_angle XX (rotate the ligand by XX degrees in X,Y,Z dimensions)</p>

## *gaussian_maps.cl*
<p>Helper script for OpenCL usage--required if using the -opencl flag, which is HIGHLY recommended for speed-up. <br>
</p>

## *ifft1.mrc* and *ETQ_0_test1.pdb*  
<p>Output files generated for visual evaluation. Note that ifft1.mrc can be visualized in Chimera. The only way I could get the grid scores properly visualized was using this kind of density map--I'm sure there's a better way though. </p>

## 2021.04.14_FFT.pdf
<p> Original presentation for Meiler lab group meeting where I explain this project and very preliminary results. 
</p>

## To do:
<p>
1. Debug step: make sure the protein/ligand are transformed appropriately. Currently when I turn the PDBs into grid representations then have to put the output back to cartesian coordinates for PDB files, the transformation might not be correct with respect to overlapping the grid scores with the output structures. <br>
<br>
2. Make FMFT-compatible. Right now, this simply loops through the XYZ dimensions, which is what FMFT addresses. <br>
<br>
3. Additional scoreterms: the FMFT method also addresses this, but I'm thinking we can do this with a different grid representation--this is something that Rosetta already has on hand and is used for low-resolution docking already as described here: <br> https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/RosettaScripts#rosettascript-sections_ligands_scoringgrids <br>
<br>
4. FFT-based docking is fast and nice in the context of high-throughput screening because the protein and ligands are represented as static structures and rely on lock-and-key type fitting. Therefore, the ultimate goal is to use this with "flexible" ligand and sidechains. My thinking for this is to represent grid regions by occupancy AKA what is the probability that this grid is occupied either given a representative ensemble of the receptor or a Dunbrack-like library for sidechains. </p>

## References
<p>
[1] Katchalski-Katzir, E., et al. (1992). "Molecular surface recognition: determination of geometric fit between proteins and their ligands by correlation techniques." Proc Natl Acad Sci U S A 89(6): 2195-2199. <br>

[2] Padhorny, D., et al. (2016). "Protein-protein docking by fast generalized Fourier transforms on 5D rotational manifolds." Proc Natl Acad Sci U S A 113(30): E4286-4293. </p>
