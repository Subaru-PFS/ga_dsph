MCMC sampler to estimate the posterior distributions of parameters that desribe the DM distribution in an observed dwarf satellite.
Authors: Tilman Hartwig & Kohei Hayashi

USAGE:
make //compile
./MCMC.exe //run model
AnalyseMCMC.jpynb //analyse results


PARAMETERS:
in main.cu
	Nchains
	Nstep
	jumpscale
	File name "out_nekoya_MCMC_%d.dat"
	DataVerr_h[i]
in functions.h
	N (number of stars that should be considered)
	THREADS_PER_BLOCK & NUM_BLOCKS (depending on N and available GPU)

Input file: mockdata/raw_Draco_core_beta_iso_1e4.csv (from Kohei Hayashi)

Output file: out_*_MCMC_*.dat
with columns: i, if_acc, LL_ratio,  LL, rh_, rhos_, rs_, alp_, bet_, gam_, rb_, net_, bet0_, betinf_, vsys_


TODO:
Velocity uncertainty (DataVerr_h) as function of S/N and stellar type.
Replace (2nd order) Jeans analysis with higher-order model
Optimize selection function for which stars should be targeted (currently first N stars in file)
Also vary velocity anisotropy
Start chains from arbitrary position (currently started at optimum)


Known issues:
"nvlink warning : Stack size for entry function ... cannot be statically determined" is normal during compilation.
