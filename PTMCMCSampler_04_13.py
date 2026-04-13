print("Using most up to date modelswitch PTMCMC")
import os
import sys
import time
import warnings

import numpy as np

from .nutsjump import HMCJump, MALAJump, NUTSJump

try:
    from mpi4py import MPI
except ImportError:
    print("Optional mpi4py package is not installed.  MPI support is not available.")
    from . import nompi4py as MPI

try:
    import acor
except ImportError:
    # Don't complain if not available.  If you set neff, you'll get an error.  Otherwise
    # it doesn't matter.
    #    print(
    #        "Optional acor package is not installed. Acor is optionally used to calculate the "
    #        "effective chain length for output in the chain file."
    #    )
    pass


def shift_array(arr, num, fill_value=0.0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class PTSampler(object):
    """
    Parallel Tempering Markov Chain Monte Carlo (PTMCMC) sampler
    
    This sampler implements adaptive jump proposals including Adaptive Metropolis (AM), Single Component Adaptive Metropolis (SCAM), and Differential Evolution (DE). Gradient-based proposals such as HMC, MALA, and NUTS can optionally be included when gradient functions are supplied
    
    Parallel tempering is implemented using MPI via ``mpi4py`` to run multiple chains at different inverse temperatures
    
    Custom proposal distributions can be added using ``addProposalToCycle``
    
    The sampler also supports **model-switching**, where the parallel tempering machinery is used to sample from mixtures of two models' posteriors
    
    Parameters
    ----------
    ndim: int
        Number of parameters in the model
    
    logl: callable or tuple of callables
        Log-likelihood function. If model-switching is enabled, this should be a tuple containing two likelihood functions
    
    logp: callable or tuple of callables
        Log-prior function. If model-switching is enabled, this should be a tuple containing two prior functions
    
    cov: array_like
        Initial covariance matrix used for adaptive proposal jumps
    
    groups: list of arrays, optional
        Parameter index groups used for adaptive covariance proposals. Each group defines a subset of parameters whose covariance structure is adapted together
    
    loglargs: list, optional
        Additional positional arguments passed to the log-likelihood
    
    loglkwargs: dict, optional
        Additional keyword arguments passed to the log-likelihood
    
    logpargs: list, optional
        Additional positional arguments passed to the log-prior
    
    logpkwargs: dict, optional
        Additional keyword arguments passed to the log-prior
    
    logl_grad: callable, optional
        Gradient of the log-likelihood function
    
    logp_grad: callable, optional
        Gradient of the log-prior function
    
    comm: MPI communicator, optional
        MPI communicator used to coordinate chains
    
    outDir: str, optional
        Directory where chain files and diagnostics are written
    
    verbose: bool, optional
        If True, print run progress to the terminal
    
    resume: bool, optional
        Resume sampling from an existing chain file
    
    seed: int, optional
        Random seed used to initialize the sampler
    """

    def __init__(
        self,
        ndim,
        logl,
        logp,
        cov,
        groups=None,
        loglargs=[],
        loglkwargs={},
        logpargs=[],
        logpkwargs={},
        logl_grad=None,
        logp_grad=None,
        comm=MPI.COMM_WORLD,
        outDir="./chains",
        verbose=True,
        resume=False,
        seed=None,
    ):
        # MPI initialization
        self.comm = comm
        self.MPIrank = self.comm.Get_rank()
        self.nchain = self.comm.Get_size()

        if self.MPIrank == 0:
            ss = np.random.SeedSequence(seed)
            child_seeds = ss.generate_state(self.nchain)
            self.stream = [np.random.default_rng(s) for s in child_seeds]
        else:
            self.stream = None
        self.stream = self.comm.scatter(self.stream, root=0)

        self.ndim = ndim

        # Infer model-switching from the *type* of logl/logp (no user-facing flag)
        logl_is_tuple = isinstance(logl, tuple)
        logp_is_tuple = isinstance(logp, tuple)

        # Must match: either both tuples (modelswitch) or both single callables (normal sampling)
        if logl_is_tuple != logp_is_tuple:
            raise ValueError(
                "Model-switching requires BOTH logl and logp to be tuples. "
                "You provided a tuple for one but not the other."
            )

        self.modelswitch = logl_is_tuple

        if self.modelswitch:
            # This code assumes exactly two models
            if len(logl) != 2 or len(logp) != 2:
                raise ValueError(
                    "For model-switching, logl and logp must be tuples of length 2."
                )

            # Keep your existing mapping (your code treats index 1 as model 1)
            self.logl1 = _function_wrapper(logl[1], loglargs, loglkwargs)
            self.logl2 = _function_wrapper(logl[0], loglargs, loglkwargs)
            self.logp1 = _function_wrapper(logp[1], logpargs, logpkwargs)
            self.logp2 = _function_wrapper(logp[0], logpargs, logpkwargs)

        else:
            self.logl = _function_wrapper(logl, loglargs, loglkwargs)
            self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        if logl_grad is not None and logp_grad is not None:
            self.logl_grad = _function_wrapper(logl_grad, loglargs, loglkwargs)
            self.logp_grad = _function_wrapper(logp_grad, logpargs, logpkwargs)
        else:
            self.logl_grad = None
            self.logp_grad = None

        self.outDir = outDir
        self.verbose = verbose
        self.resume = resume

        # setup output file
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass

        # find indices for which to perform adaptive jumps
        self.groups = groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]

        # set up covariance matrix
        self.cov = cov
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]

            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

        self.M2 = np.zeros((ndim, ndim))
        self.mu = np.zeros(ndim)

        # initialize proposal cycle
        self.propCycle = []
        self.jumpDict = {}

        # indicator for auxilary jumps
        self.aux = []

    def initialize(
        self,
        Niter,
        ladder=None,
        shape="geometric",
        Bmax=1,
        Bmin=None,
        Tmin=None,
        Tmax=None,
        Tskip=100,
        isave=1000,
        covUpdate=1000,
        SCAMweight=30,
        AMweight=20,
        DEweight=50,
        NUTSweight=20,
        HMCweight=20,
        MALAweight=0,
        burn=50000,
        HMCstepsize=0.1,
        HMCsteps=300,
        maxIter=None,
        thin=10,
        i0=0,
        neff=None,
        writeHotChains=False,
        hotChain=False,
        beta_schedule=None,
        hold_iter=0,
        nameChainTemps=False
    ):
        """
        Initialize MCMC quantities
        
        Parameters
        ----------
        Niter: int
            Number of iterations for the cold chain (beta = 1) when using standard PTMCMC

        ladder: array-like, optional
            User-defined temperature or beta ladder. Either temperatures or inverse temperatures (betas) may be supplied

        shape: {"geometric", "linear"}, optional
            Shape of the automatically generated temperature/beta ladder when `ladder` is not supplied. Ignored if `ladder` or `beta_schedule` is provided

        Bmax: float
            Maximum beta value (default = 1)

        Bmin: float, optional
            Minimum beta value

        Tmin, Tmax: float, optional
            Alternative temperature specification for ladder construction

        Tskip: int
            Number of iterations between parallel-tempering swap attempts

        isave: int
            Number of iterations between writing samples to disk

        covUpdate: int
            Number of iterations between adaptive covariance updates

        SCAMweight, AMweight, DEweight: int
            Relative weights of SCAM, AM, and Differential Evolution jump proposals

        NUTSweight, HMCweight, MALAweight: int
            Relative weights of gradient-based jump proposals

        burn: int
            Burn-in iterations before enabling Differential Evolution jumps

        HMCstepsize: float
            Step size for HMC proposals

        HMCsteps: int
            Maximum trajectory length for HMC proposals

        maxIter: int, optional
            Maximum number of iterations for high-temperature chains

        thin: int
            Thinning interval for storing samples

        i0: int
            Initial iteration index when resuming runs

        neff: int, optional
            Target effective sample size for early termination

        writeHotChains: bool
            If True, write hot chains to disk

        hotChain: bool
            If True, include a beta=0 hot chain

        beta_schedule: array-like, optional
            Optional schedule of inverse temperatures (betas) to apply at each iteration. If provided, parallel tempering is disabled and the chain runs as a single chain whose beta changes over time

            Values must lie in the interval [0, 1]

            This is typically used for thermodynamic integration or power-posterior sampling, where the likelihood contribution is gradually turned on

        hold_iter: int, optional
            Number of initial iterations for which beta is fixed to 0 before following `beta_schedule`. This effectively prepends a plateau of beta = 0 values to the schedule

        nameChainTemps: bool
            If True, chain files are named using temperatures instead of betas

        Notes
        -----
        When `beta_schedule` is provided:

        * Parallel tempering is disabled
        * Only single-chain runs are supported
        * `ladder` and `hotChain` options are not allowed
        * The number of iterations is determined by the schedule length
        """

        # Varying-beta schedule (power posterior / thermodynamic integration mode)
        self.beta_schedule = None
        self.disable_pt = False
        scheduling_active = beta_schedule is not None

        if hold_iter < 0:
            raise ValueError("hold_iter must be >= 0")

        if scheduling_active:
            if hotChain:
                raise ValueError("hotChain is not compatible with beta_schedule runs")
            if ladder is not None:
                raise ValueError(
                    f"beta_schedule is not compatible with ladder={ladder}. Omit ladder for beta_schedule runs."
                )
            if self.nchain > 1:
                raise ValueError(
                    f"beta_schedule is only supported for single-chain runs, but MPI size is {self.nchain}. "
                    "Run without MPI or use a single MPI process."
                )
    
            self.disable_pt = True
    
            # Parse beta_schedule as array-like custom schedule
            beta_core = np.asarray(beta_schedule, dtype=float).reshape(-1)
                
            hold = np.zeros(int(hold_iter), dtype=float)
            full = np.concatenate([hold, beta_core])
    
            if full.ndim != 1 or full.size == 0:
                raise ValueError("beta_schedule produced an empty schedule")
            if not np.all(np.isfinite(full)):
                raise ValueError("beta_schedule contains non-finite values")
            if np.min(full) < 0.0 or np.max(full) > 1.0:
                raise ValueError("beta_schedule values must lie in [0, 1]")
    
            self.beta_schedule = full
            self.beta = float(full[0])

            # Warn if thinning skips beta schedule points
            if thin != 1:
                n_total = len(full)
                n_used = (n_total + thin - 1) // thin  # ceil division
            
                percent = 100.0 * n_used / n_total
            
                warnings.warn(
                    f"[beta_schedule] thin={thin} → using {percent:.1f}% "
                    f"of beta grid ({n_used}/{n_total} points retained)",
                    RuntimeWarning
                )
    
            # Override Niter/maxIter based on schedule, full.size is the number of beta values for stored states, including the initial state, so the number of transition steps is one less. 
            Niter = int(full.size) - 1
            if Niter < 0:
                raise ValueError("beta_schedule must contain at least one value")
            if maxIter is None:
                maxIter = Niter

        # Default maxIter for non-scheduled runs
        if maxIter is None:
            maxIter = Niter

        if isave % thin != 0:
            raise ValueError("isave = %d is not a multiple of thin =  %d" % (isave, thin))

        if Niter % thin != 0:
            print(
                "Niter = %d is not a multiple of thin = %d.  The last %d samples will be lost"
                % (Niter, thin, Niter % thin)
            )

        self.ladder = ladder
        self.covUpdate = covUpdate
        self.SCAMweight = SCAMweight
        self.AMweight = AMweight
        self.DEweight = DEweight
        self.burn = burn
        self.Tskip = Tskip
        self.thin = thin
        self.isave = isave
        self.Niter = Niter
        self.neff = neff
        self.tstart = 0
            
        # Output/resume format flag (preserve legacy format unless feature is active)
        self.write_beta_col = (self.beta_schedule is not None)

        N = int(maxIter / thin) + 1  # first sample + those we generate

        self._lnprob = np.zeros(N)
        self._lnlike = np.zeros(N)
        self._chain = np.zeros((N, self.ndim))
        self._beta = np.zeros(N)
        self.ind_next_write = 0  # Next index in these arrays to write out
        self.naccepted = 0
        self.swapProposed = 0
        self.nswap_accepted = 0

        self.n_metaparams = 8 if self.modelswitch else 4

        if self.modelswitch:
            self._lnprob1 = np.zeros(N)
            self._lnlike1 = np.zeros(N)
            self._lnprob2 = np.zeros(N)
            self._lnlike2 = np.zeros(N)

        # set up covariance matrix and DE buffers
        if self.MPIrank == 0:
            self._AMbuffer = np.zeros((self.covUpdate, self.ndim))
            self._DEbuffer = np.zeros((self.burn, self.ndim))

        # ##### setup default jump proposal distributions ##### #

        # Gradient-based jumps
        if self.logl_grad is not None and self.logp_grad is not None:
            # DOES MALA do anything with the burnin? (Not adaptive enabled yet)
            malajump = MALAJump(self.logl_grad, self.logp_grad, self.cov, self.burn)
            self.addProposalToCycle(malajump, MALAweight)
            if MALAweight > 0:
                print("WARNING: MALA jumps are not working properly yet")

            # Perhaps have an option to adaptively tune the mass matrix?
            # Now that is done by defaulk
            hmcjump = HMCJump(
                self.logl_grad,
                self.logp_grad,
                self.cov,
                self.burn,
                stepsize=HMCstepsize,
                nminsteps=2,
                nmaxsteps=HMCsteps,
            )
            self.addProposalToCycle(hmcjump, HMCweight)

            # Target acceptance rate (delta) should be optimal for 0.6
            nutsjump = NUTSJump(
                self.logl_grad,
                self.logp_grad,
                self.cov,
                self.burn,
                trajectoryDir=None,
                write_burnin=False,
                force_trajlen=None,
                force_epsilon=None,
                delta=0.6,
            )
            self.addProposalToCycle(nutsjump, NUTSweight)

        # add SCAM
        self.addProposalToCycle(self.covarianceJumpProposalSCAM, self.SCAMweight)

        # add AM
        self.addProposalToCycle(self.covarianceJumpProposalAM, self.AMweight)

        # check length of jump cycle
        if len(self.propCycle) == 0:
            raise ValueError("No jump proposals specified!")

        # randomize cycle
        self.randomizeProposalCycle()

        # Ladder setup
        if scheduling_active:
            # varying-beta run: no PT ladder
            self.ladder = np.array([1.0])
        else:
            # if ladder given check if in temp or beta
            if self.ladder is not None and len(self.ladder) > 0:
                if max(self.ladder) > 1:
                    # user gave temperatures >>> convert to beta
                    self.ladder = [1 / temp for temp in self.ladder]

            # ladder not specified, create one
            else:
                # If temperatures are used, convert to beta
                if Tmin:  # used temperatures
                    Bmax = 1 / Tmin  # Tmin is typically 1
                if Tmax:
                    Bmin = 1 / Tmax

                self.ladder = self.Ladder(Bmax, Bmin=Bmin, shape=shape)

            # beta for current chain (only meaningful for PT runs)
            self.beta = self.ladder[self.MPIrank]

        # Name chain files
        if scheduling_active:
            # beta changes over time, fixed filename for scheduled runs
            self.fname = self.outDir + "/chain_schedule.txt"
        else:
            if hotChain and self.MPIrank == self.nchain - 1:
                self.beta = 0  # This is the "hot chain"
                if nameChainTemps:  # if you prefer the old naming scheme
                    self.fname = self.outDir + "/chain_hot.txt"
                else:  # new naming scheme with beta
                    self.fname = self.outDir + "/chain_0.txt"

            elif nameChainTemps:  # if you prefer the old naming scheme
                self.fname = self.outDir + "/chain_{0}.txt".format(1 / self.beta)

            else:  # new naming scheme with beta
                self.fname = self.outDir + "/chain_{0}.txt".format(self.beta)

        # write hot chains
        self.writeHotChains = writeHotChains
    
        self.resumeLength = 0
        if self.resume and os.path.isfile(self.fname):
            if self.verbose:
                print("Resuming run from chain file {0}".format(self.fname))
            try:
                self.resumechain = np.loadtxt(self.fname, ndmin=2)
                expected_cols = (1 if self.write_beta_col else 0) + self.ndim + self.n_metaparams
                if self.resumechain.shape[1] != expected_cols:
                    current_mode = "beta_schedule=True" if self.write_beta_col else "beta_schedule=False"
                    raise Exception(
                        f"Cannot resume chain file {self.fname}: expected {expected_cols} columns for {current_mode}, "
                        f"but found {self.resumechain.shape[1]}. "
                        "This usually means the chain file was created with a different resume/output format."
                    )
                self.resumeLength = self.resumechain.shape[0]  # Number of samples read from old chain
            except ValueError as error:
                print("Reading old chain files failed with error", error)
                raise Exception("Couldn't read old chain to resume")
            self._chainfile = open(self.fname, "a")
            if (
                self.isave != self.thin  # This special case is always OK
                and self.resumeLength % (self.isave / self.thin) != 1  # Initial sample plus blocks of isave/thin
            ):
                raise Exception(
                    (
                        "Old chain has {0} rows, which is not the initial sample plus a multiple of isave/thin = {1}"
                    ).format(self.resumeLength, self.isave // self.thin)
                )
            print(
                "Resuming with",
                self.resumeLength,
                "samples from file representing",
                (self.resumeLength - 1) * self.thin + 1,
                "original samples",
            )
        else:
            self._chainfile = open(self.fname, "w")
        self._chainfile.close()

    def updateChains(
        self,
        p0,
        lnlike0,
        lnprob0,
        iter,
        lnlike1=None,
        lnprob1=None,
        lnlike2=None,
        lnprob2=None,
    ):
        """
        
        Update internal chain storage after each accepted or rejected proposal
        
        Stores parameter values, log-likelihood, log-posterior, and current beta into the sampler buffers. Data are written to disk periodically according to the `isave` setting

        """
        # update buffer
        if self.MPIrank == 0:
            self._AMbuffer[iter % self.covUpdate, :] = p0

        # put results into arrays
        if iter % self.thin == 0:
            ind = int(iter / self.thin)
            self._chain[ind, :] = p0
            self._beta[ind] = self.beta
            self._lnlike[ind] = lnlike0
            self._lnprob[ind] = lnprob0

            if (lnlike1 is not None) and (lnlike2 is not None) and (lnprob1 is not None) and (lnprob2 is not None):
                self._lnlike1[ind] = lnlike1
                self._lnprob1[ind] = lnprob1
                self._lnlike2[ind] = lnlike2
                self._lnprob2[ind] = lnprob2

        # write to file
        if iter % self.isave == 0:
            self.writeOutput(iter)

    def writeOutput(self, iter):
        """
        Write chains and covariance matrix.  Called every isave on samples or at end.
        """
        if iter // self.thin >= self.ind_next_write:

            if self.writeHotChains or self.MPIrank == 0:
                self._writeToFile(iter)

            # write output covariance matrix
            if iter > 0:
                np.save(self.outDir + "/cov.npy", self.cov)

            if self.MPIrank == 0 and self.verbose:
                if iter > 0:
                    sys.stdout.write("\r")
                percent = iter / self.Niter * 100  # Percent of total work finished
                acceptance = self.naccepted / iter if iter > 0 else 0
                elapsed = time.time() - self.tstart
                if self.resume:
                    # Percentage of new work done
                    percentnew = (
                        (iter - self.resumeLength * self.thin) / (self.Niter - self.resumeLength * self.thin) * 100
                    )
                    sys.stdout.write(
                        "Finished %2.2f percent (%2.2f percent of new work) in %f s Acceptance rate = %g"
                        % (percent, percentnew, elapsed, acceptance)
                    )
                else:
                    sys.stdout.write(
                        "Finished %2.2f percent in %f s Acceptance rate = %g" % (percent, elapsed, acceptance)
                    )
                sys.stdout.flush()

    def sample(
        self,
        p0,
        Niter,
        Bmax=1,
        Bmin=None,
        ladder=None,
        shape="geometric",
        Tmin=None,
        Tmax=None,
        Tskip=100,
        isave=1000,
        covUpdate=1000,
        SCAMweight=20,
        AMweight=20,
        DEweight=20,
        NUTSweight=20,
        MALAweight=20,
        HMCweight=20,
        burn=10000,
        HMCstepsize=0.1,
        HMCsteps=300,
        maxIter=None,
        thin=10,
        i0=0,
        neff=None,
        writeHotChains=False,
        hotChain=False,
        beta_schedule=None,
        hold_iter=0,
        nameChainTemps=False,
    ):
        """
        Run the PTMCMC sampler. This function performs Parallel Tempering Markov Chain Monte Carlo sampling. Depending on configuration, the sampler operates in one of two modes:
        1. Standard Parallel Tempering (default)
        2. Varying-beta / power-posterior sampling (when ``beta_schedule`` is provided)
        
        Parameters
        ----------
        p0: array_like
            Initial parameter vector
        
        Niter: int
            Number of iterations for the cold chain (beta = 1) in standard PTMCMC runs
        
        Bmax: float, optional
            Maximum beta value (default = 1)
        
        Bmin: float, optional
            Minimum beta value in the ladder
        
        ladder: array_like, optional
            User-specified beta or temperature ladder. If temperatures are supplied, they will automatically be converted to betas
        
        shape: {"geometric", "linear"}, optional
            Shape of the automatically generated temperature/beta ladder when ``ladder`` is not provided. Ignored if ``ladder`` or ``beta_schedule`` is supplied
        
        Tmin: float, optional
            Minimum temperature used to construct the ladder
        
        Tmax: float, optional
            Maximum temperature used to construct the ladder
        
        Tskip: int, optional
            Number of iterations between proposed temperature swaps
        
        isave: int, optional
            Number of iterations between writing samples to disk
        
        covUpdate: int, optional
            Number of iterations between adaptive covariance updates
        
        SCAMweight: int, optional
            Weight of the SCAM jump proposal in the proposal cycle
        
        AMweight: int, optional
            Weight of the Adaptive Metropolis jump proposal
        
        DEweight: int, optional
            Weight of the Differential Evolution jump proposal
        
        NUTSweight: int, optional
            Weight of the No-U-Turn Sampler proposal
        
        MALAweight: int, optional
            Weight of the MALA proposal
        
        HMCweight: int, optional
            Weight of the Hamiltonian Monte Carlo proposal
        
        burn: int, optional
            Burn-in period before Differential Evolution jumps are enabled
        
        HMCstepsize: float, optional
            Step size used in Hamiltonian Monte Carlo proposals
        
        HMCsteps: int, optional
            Maximum number of leapfrog steps used in HMC trajectories
        
        maxIter: int, optional
            Maximum number of iterations allowed for high-temperature chains. Defaults to ``2 * Niter`` in standard PT runs
        
        thin: int, optional
            Thinning interval for recorded samples
        
        i0: int, optional
            Initial iteration index. Used when resuming a previous run
        
        neff: int, optional
            Target number of effective samples before stopping early
        
        writeHotChains: bool, optional
            If True, write hot chains to disk
        
        hotChain: bool, optional
            If True, include a beta = 0 chain
        
        beta_schedule: array_like, optional
            Sequence of beta values to apply at each iteration. When provided, the sampler runs in varying-beta mode rather than standard parallel tempering
        
            This is typically used for thermodynamic integration or power-posterior sampling, where the likelihood contribution is gradually introduced by increasing beta from 0 to 1
        
            Values must lie in the interval [0, 1]
        
            When ``beta_schedule`` is used:
                - Parallel tempering swaps are disabled
                - Only single-chain runs are supported
                - ``ladder`` and ``hotChain`` options cannot be used
                - The total number of iterations is determined by the schedule
                  length
        
        hold_iter: int, optional
            Number of initial iterations to hold beta = 0 before following the provided ``beta_schedule``. This effectively prepends a plateau of beta = 0 values to the schedule
        
        nameChainTemps: bool, optional
            If True, chain files are named using temperatures instead of betas
        
        Notes
        -----
        If ``beta_schedule`` is supplied, the sampler performs a varying-beta run where the inverse temperature changes deterministically at each iteration rather than using a parallel tempering ladder
        """       

        # set up arrays to store lnprob, lnlike and chain
        # if picking up from previous run, don't re-initialize
        if i0 == 0:
            self.initialize(
                Niter,
                Bmax=Bmax,
                Bmin=Bmin,
                ladder=ladder,
                shape=shape,
                Tmin=Tmin,
                Tmax=Tmax,
                Tskip=Tskip,
                isave=isave,
                covUpdate=covUpdate,
                SCAMweight=SCAMweight,
                AMweight=AMweight,
                DEweight=DEweight,
                NUTSweight=NUTSweight,
                MALAweight=MALAweight,
                HMCweight=HMCweight,
                burn=burn,
                HMCstepsize=HMCstepsize,
                HMCsteps=HMCsteps,
                maxIter=maxIter,
                thin=thin,
                i0=i0,
                neff=neff,
                writeHotChains=writeHotChains,
                beta_schedule=beta_schedule,
                hold_iter=hold_iter,
                hotChain=hotChain,
                nameChainTemps=nameChainTemps
            )

        # compute lnprob for initial point in chain... if resuming, start from the LAST saved point in the chain
        if self.resume and self.resumeLength > 0:

            last_row = self.resumeLength - 1
            param_start = 1 if self.write_beta_col else 0

            if self.write_beta_col:
                self.beta = self.resumechain[last_row, 0]
            else:
                # legacy format  beta comes from ladder
                self.beta = self.ladder[self.MPIrank]

            p0 = self.resumechain[last_row, param_start : param_start + self.ndim]

            lnprob0 = self.resumechain[last_row, -self.n_metaparams]
            lnlike0 = self.resumechain[last_row, -(self.n_metaparams - 1)]

            if self.modelswitch:
                lnprob1 = self.resumechain[last_row, -(self.n_metaparams - 2)]
                lnlike1 = self.resumechain[last_row, -(self.n_metaparams - 3)]
                lnprob2 = self.resumechain[last_row, -(self.n_metaparams - 4)]
                lnlike2 = self.resumechain[last_row, -(self.n_metaparams - 5)]

            self.ind_next_write = self.resumeLength
            self.naccepted = int(round(((self.resumeLength - 1) * self.thin) * self.resumechain[last_row, -2]))
            i0 = (self.resumeLength - 1) * self.thin

        else:
            # compute prior and likelihood
            if not self.modelswitch:
                lp = self.logp(p0)

                if lp == -np.inf:
                    lnlike0 = -np.inf
                    lnprob0 = -np.inf

                else:
                    lnlike0 = self.logl(p0)
                    lnprob0 = self.beta * lnlike0 + lp

            elif self.modelswitch:  # Using modelswitch
                
                lp1 = self.logp1(p0)
                lp2 = self.logp2(p0)

                if lp1 == -np.inf or lp2 == -np.inf:
                    lnprob0 = -np.inf
                    lnlike0 = -np.inf
                    lnlike1 = -np.inf
                    lnprob1 = -np.inf
                    lnlike2 = -np.inf
                    lnprob2 = -np.inf

                else:
                    lnlike1 = self.logl1(p0) 
                    lnprob1 = lnlike1 + lp1

                    lnlike2 = self.logl2(p0)
                    lnprob2 = lnlike2 + lp2

                    lnlike0 = lnprob1 - lnprob2  # Not power posteriors

                    lnprob0 = self.beta * (lnlike0) + lnprob2  #

        # If beta_schedule is active, make the current state consistent with the schedule
        if getattr(self, "beta_schedule", None) is not None:
            current_idx = i0
            if current_idx < 0 or current_idx >= len(self.beta_schedule):
                raise IndexError(
                    f"beta_schedule index out of range at initialization: idx={current_idx}, len={len(self.beta_schedule)}"
                )

            self.beta = float(self.beta_schedule[current_idx])

            if not self.modelswitch:
                lp0 = self.logp(p0)
                if lp0 == -np.inf or (not np.isfinite(lnlike0)):
                    lnprob0 = -np.inf
                else:
                    lnprob0 = self.beta * lnlike0 + lp0
            else:
                if (lnprob2 is None) or (not np.isfinite(lnprob2)) or (not np.isfinite(lnlike0)):
                    lnprob0 = -np.inf
                else:
                    lnprob0 = self.beta * lnlike0 + lnprob2
        
        if not self.modelswitch:
            # record first values
            self.updateChains(p0, lnlike0, lnprob0, i0)

        elif self.modelswitch:  # Using modelswitch
            # record first values
            self.updateChains(
                p0,
                lnlike0,
                lnprob0,
                i0,
                lnlike1=lnlike1,
                lnprob1=lnprob1,
                lnlike2=lnlike2,
                lnprob2=lnprob2,
            )

        self.comm.barrier()
        self.tstart = time.time()

        # start iterations
        iter = i0
       
        runComplete = False
        while runComplete is False:
            iter += 1
            self.comm.barrier()  # make sure all processes are at the same iteration
            # call PTMCMCOneStep
            if not self.modelswitch:
                p0, lnlike0, lnprob0 = self.PTMCMCOneStep(p0, lnlike0, lnprob0, iter)
            elif self.modelswitch:
                p0, lnlike0, lnprob0, lnlike1, lnprob1, lnlike2, lnprob2 = self.PTMCMCOneStep(
                    p0,
                    lnlike0,
                    lnprob0,
                    iter,
                    lnlike1=lnlike1,
                    lnprob1=lnprob1,
                    lnlike2=lnlike2,
                    lnprob2=lnprob2,
                )

            # rank 0 decides whether to stop
            if self.MPIrank == 0:
                if iter >= self.Niter:  # stop if reached maximum number of iterations
                    message = "\nRun Complete"
                    runComplete = True
                elif self.neff:  # Stop if effective number of samples reached if requested
                    if iter % 1000 == 0 and iter > 2 * self.burn and self.MPIrank == 0:
                        Neff = iter / max(
                            1,
                            np.nanmax(
                                [acor.acor(self._chain[self.burn : (iter - 1), ii])[0] for ii in range(self.ndim)]
                            ),
                        )
                        # print('\n {0} effective samples'.format(Neff))
                        if int(Neff) >= self.neff:
                            message = "\nRun Complete with {0} effective samples".format(int(Neff))
                            runComplete = True

            runComplete = self.comm.bcast(runComplete, root=0)  # rank 0 tells others whether to stop

            if runComplete:
                self.writeOutput(iter)  # Possibly write partial block
                if self.MPIrank == 0 and self.verbose:
                    print(message)

    def PTMCMCOneStep(
        self,
        p0,
        lnlike0,
        lnprob0,
        iter,
        lnlike1=None,
        lnprob1=None,
        lnlike2=None,
        lnprob2=None,
    ):
        """
        Perform a single PTMCMC iteration
        
        Parameters
        ----------
        p0: ndarray
            Current parameter vector
        
        lnlike0: float
            Current log-likelihood value
        
        lnprob0: float
            Current log-posterior value
        
        iter: int
            Current iteration index
        
        lnlike1, lnprob1, lnlike2, lnprob2 : float, optional
            Model-switch likelihood and posterior values when model-switching mode is enabled
        
        Returns
        -------
        p0: ndarray
            Updated parameter vector
        
        lnlike0: float
            Updated log-likelihood value
        
        lnprob0: float
            Updated log-posterior value
        """
        # update covariance matrix
        if (iter - 1) % self.covUpdate == 0 and (iter - 1) != 0 and self.MPIrank == 0:
            self._updateRecursive(iter - 1, self.covUpdate)

            # broadcast to other chains
            [self.comm.send(self.cov, dest=rank + 1, tag=111) for rank in range(self.nchain - 1)]

        # update covariance matrix
        if (iter - 1) % self.covUpdate == 0 and (iter - 1) != 0 and self.MPIrank > 0:
            self.cov[:, :] = self.comm.recv(source=0, tag=111)
            for ct, group in enumerate(self.groups):
                covgroup = np.zeros((len(group), len(group)))
                for ii in range(len(group)):
                    for jj in range(len(group)):
                        covgroup[ii, jj] = self.cov[group[ii], group[jj]]

                self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

        # update DE buffer
        if (iter - 1) % self.burn == 0 and (iter - 1) != 0 and self.MPIrank == 0:
            self._updateDEbuffer(iter - 1, self.burn)

            # broadcast to other chains
            [self.comm.send(self._DEbuffer, dest=rank + 1, tag=222) for rank in range(self.nchain - 1)]

        # update DE buffer
        if (iter - 1) % self.burn == 0 and (iter - 1) != 0 and self.MPIrank > 0:
            self._DEbuffer = self.comm.recv(source=0, tag=222)

            # randomize cycle
            if self.DEJump not in self.propCycle:
                self.addProposalToCycle(self.DEJump, self.DEweight)
                self.randomizeProposalCycle()

        # after burn in, add DE jumps
        if (iter - 1) == self.burn and self.MPIrank == 0:
            if self.verbose:
                print("Adding DE jump with weight {0}".format(self.DEweight))
            self.addProposalToCycle(self.DEJump, self.DEweight)

            # randomize cycle
            self.randomizeProposalCycle()
        
        # Recompute lnprob0 under current beta
        if getattr(self, "beta_schedule", None) is not None:
            idx = iter
            if idx < 0 or idx >= len(self.beta_schedule):
                raise IndexError(
                    f"beta_schedule index out of range: idx={idx}, len={len(self.beta_schedule)}"
                )
            self.beta = float(self.beta_schedule[idx])

            if not self.modelswitch:
                lp0 = self.logp(p0)
                if lp0 == -np.inf or not np.isfinite(lnlike0):
                    lnprob0 = -np.inf
                else:
                    lnprob0 = self.beta * lnlike0 + lp0
            else:
                if (lnprob2 is None) or (not np.isfinite(lnprob2)) or (not np.isfinite(lnlike0)):
                    lnprob0 = -np.inf
                else:
                    lnprob0 = self.beta * lnlike0 + lnprob2
        
        # jump proposal, once sample() has restored the last saved state, resume proceeds normally
        y, qxy, jump_name = self._jump(p0, iter)  # made a jump
        self.jumpDict[jump_name][0] += 1

        # compute prior and likelihood
        if not self.modelswitch:
            lp = self.logp(y)

            if lp == -np.inf:
                newlnlike = -np.inf
                newlnprob = -np.inf

            else:
                newlnlike = self.logl(y)
                newlnprob = self.beta * newlnlike + lp

        elif self.modelswitch:  # Using modelswitch

            lp1 = self.logp1(y)
            lp2 = self.logp2(y)

            if lp1 == -np.inf or lp2 == -np.inf:
                newlnlike = -np.inf
                newlnprob = -np.inf
                newlnlike1 = -np.inf
                newlnprob1 = -np.inf
                newlnlike2 = -np.inf
                newlnprob2 = -np.inf

            else:
                newlnlike1 = self.logl1(y)
                newlnprob1 = newlnlike1 + lp1  # no beta here, we want full posterior of each model

                newlnlike2 = self.logl2(y)
                newlnprob2 = newlnlike2 + lp2  # no beta here, we want full posterior of each model

                newlnlike = newlnprob1 - newlnprob2

                # ln posterior = beta * ln likelihood + ln prior
                # ln prior is set to ln posterior of the second model
                # ln likelihood is the difference between ln posterior of the first and second models
                # beta determines how much of newlnprob1 vs newlnprob2
                newlnprob = self.beta * (newlnlike) + newlnprob2

        # hastings step
        diff = newlnprob - lnprob0 + qxy

        rand_log = np.log(self.stream.random())
        if diff > rand_log:
                
            # accept jump
            p0, lnlike0, lnprob0 = y, newlnlike, newlnprob
            if self.modelswitch:
                lnlike1, lnlike2, lnprob1, lnprob2 = (
                    newlnlike1,
                    newlnlike2,
                    newlnprob1,
                    newlnprob2,
                )

            # update acceptance counter
            self.naccepted += 1
            self.jumpDict[jump_name][1] += 1

        # Update chains
        if self.modelswitch:
            self.updateChains(
                p0,
                lnlike0,
                lnprob0,
                iter,
                lnlike1=lnlike1,
                lnprob1=lnprob1,
                lnlike2=lnlike2,
                lnprob2=lnprob2,
            )
            return p0, lnlike0, lnprob0, lnlike1, lnprob1, lnlike2, lnprob2

        else:
            # temperature swap
            if (not getattr(self, "disable_pt", False)) and (iter % self.Tskip == 0) and (self.nchain > 1):
                p0, lnlike0, lnprob0 = self.PTswap(p0, lnlike0, lnprob0, iter)

            self.updateChains(p0, lnlike0, lnprob0, iter)

            return p0, lnlike0, lnprob0

    def PTswap(self, p0, lnlike0, lnprob0, iter):
        """
        Parallel tempering swap using betas.
        Assumes self.ladder is an array of betas (ladder[0] ~ 1, decreasing).
        """

        betas = np.asarray(self.ladder, dtype=float)  # betas per chain index

        # Gather current states/likelihoods to rank 0
        log_Ls = self.comm.gather(lnlike0, root=0)
        p0s = self.comm.gather(p0, root=0)

        # Prepare per-swap acceptance bookkeeping
        swap_accepted = np.zeros(self.nchain, dtype=float)

        new_p0s = None
        new_log_Ls = None

        if self.MPIrank == 0:
            # Buffers to scatter back out
            new_p0s = [None] * self.nchain
            new_log_Ls = [None] * self.nchain

            # swap_map maps "chain index" -> "which gathered state it currently holds"
            swap_map = list(range(self.nchain))

            # Propose swaps between adjacent chains, from hot end toward cold end
            for i in reversed(range(self.nchain - 1)):
                a = swap_map[i]       # index of state currently at chain i
                b = swap_map[i + 1]   # index of state currently at chain i+1

                # log alpha = (beta_i - beta_{i+1}) * (L_b - L_a)
                log_alpha = (betas[i] - betas[i + 1]) * (log_Ls[b] - log_Ls[a])

                if np.log(self.stream.random()) < log_alpha:
                    swap_map[i], swap_map[i + 1] = swap_map[i + 1], swap_map[i]
                    swap_accepted[i] += 1.0

            # Build the swapped arrays for scatter
            for j in range(self.nchain):
                new_p0s[j] = p0s[swap_map[j]]
                new_log_Ls[j] = log_Ls[swap_map[j]]

        # scatter back swapped lists (rank 0 provides them; others can pass None safely, but
        # passing the original gathered lists is the most robust convention)
        p0 = self.comm.scatter(new_p0s if self.MPIrank == 0 else None, root=0)
        lnlike0 = self.comm.scatter(new_log_Ls if self.MPIrank == 0 else None, root=0)

        # Track acceptance stats
        self.nswap_accepted += self.comm.scatter(swap_accepted, root=0)
        self.swapProposed += 1

        # Recompute posterior for this chain under its beta
        lnprob0 = self.beta * lnlike0 + self.logp(p0)

        return p0, lnlike0, lnprob0

    def Ladder(self, Bmax, Bmin=None, tstep=None, shape="geometric"):
        """
        Method to compute temperature/beta ladder. The default is a geometrically
        spaced ladder with a spacing designed to give 25 % temperature/beta swap
        acceptance rate. The other option is a linear spacing.

        """

        # TODO: make options to do other temperature ladders

        if self.nchain > 1:
            if shape == "linear":
                if tstep is None and Bmin is None:  # Bmin set to 0
                    if Bmin is None:
                        warnings.warn("Bmin not given. Bmin will be set to 0 for linear spacing.")
                        Bmin = 0
                    tstep = Bmax / (self.nchain - 1)

                elif tstep is None and Bmin is not None:
                    tstep = (Bmax - Bmin) / (self.nchain - 1)

                ladder = np.zeros(self.nchain)
                for ii in range(self.nchain):
                    ladder[ii] = round(Bmax - (tstep * ii), 5)

            if shape == "geometric":
                if tstep is None and Bmin is None:
                    tstep = 1 + np.sqrt(2 / self.ndim)

                elif tstep is None and Bmin is not None:
                    if Bmin == 0:
                        warnings.warn(
                            "Bmin set to 0. Geometric series can only approach beta=0. Make sure to include the"
                            "hot chain to get a beta=0 chain if you haven't already. Bmin will be set to 1e-7."
                        )
                        Bmin = 1e-7
                    tstep = np.exp(np.log(Bmin / Bmax) / (1 - self.nchain))  # Bmin can't be 0 here

                ladder = np.zeros(self.nchain)
                for ii in range(self.nchain):
                    ladder[ii] = Bmax * tstep ** (-ii)
        else:
            ladder = np.array([Bmax])

        return ladder

    def _writeToFile(self, iter):
        """
        Write sampler output to the chain file
        
        Each row written contains the parameter vector followed by diagnostic quantities including the log-posterior, log-likelihood, acceptance rate, and parallel-tempering acceptance rate
        
        In model-switching mode, additional columns are written for the log-posterior and log-likelihood of each model
        
        Parameters
        ----------
        iter: int
            Current iteration number

        """

        self._chainfile = open(self.fname, "a+")
        # index 0 is the initial element.  So after 10*thin iterations we need to write elements 1..10
        write_end = iter // self.thin + 1  # First element not to write.
        for ind in range(self.ind_next_write, write_end):
            pt_acc = 1
            if self.MPIrank < self.nchain - 1 and self.swapProposed != 0:
                pt_acc = self.nswap_accepted / self.swapProposed

            # beta column only for varying-beta runs
            if self.write_beta_col:
                self._chainfile.write("%f\t" % self._beta[ind])

            # then parameters (always)
            self._chainfile.write(
                "\t".join(["%22.22f" % (self._chain[ind, kk]) for kk in range(self.ndim)])
            )

            # main posterior / likelihood for the active chain state
            self._chainfile.write(
                "\t%f\t%f" % (self._lnprob[ind], self._lnlike[ind])
            )

            # extra model-switch diagnostics, if present
            if self.modelswitch:
                self._chainfile.write(
                    "\t%f\t%f\t%f\t%f"
                    % (
                        self._lnprob1[ind],
                        self._lnlike1[ind],
                        self._lnprob2[ind],
                        self._lnlike2[ind],
                    )
                )

            # acceptance metadata goes last
            self._chainfile.write(
                "\t%f\t%f\n" % (self.naccepted / iter if iter > 0 else 0, pt_acc)
            )
        self._chainfile.close()
        self.ind_next_write = write_end  # Ready for next write

        # write jump statistics files ####

        # only for T=1 chain
        if self.MPIrank == 0:

            # first write file contaning jump names and jump rates
            fout = open(self.outDir + "/jumps.txt", "w")
            njumps = len(self.propCycle)
            ujumps = np.array(list(set(self.propCycle)))
            for jump in ujumps:
                fout.write("%s %4.2g\n" % (jump.__name__, np.sum(np.array(self.propCycle) == jump) / njumps))

            fout.close()

            # now write jump statistics for each jump proposal
            for jump in self.jumpDict:
                fout = open(self.outDir + "/" + jump + "_jump.txt", "a+")
                fout.write("%g\n" % (self.jumpDict[jump][1] / max(1, self.jumpDict[jump][0])))
                fout.close()

    # function to update covariance matrix for jump proposals
    def _updateRecursive(self, iter, mem):
        """
        Recursively update the adaptive covariance matrix used for proposal jumps
        
        Parameters
        ----------
        iter: int
            Current iteration index
        
        mem: int
            Number of samples used in the covariance update window
        """
        it = iter - mem
        ndim = self.ndim

        if it == 0:
            self.M2 = np.zeros((ndim, ndim))
            self.mu = np.zeros(ndim)

        for ii in range(mem):
            diff = np.zeros(ndim)
            it += 1
            for jj in range(ndim):

                diff[jj] = self._AMbuffer[ii, jj] - self.mu[jj]
                self.mu[jj] += diff[jj] / it

            self.M2 += np.outer(diff, (self._AMbuffer[ii, :] - self.mu))

        self.cov[:, :] = self.M2 / (it - 1)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]

            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

    # update DE buffer samples
    def _updateDEbuffer(self, iter, burn):
        """
        Update the Differential Evolution proposal buffer
        
        The DE buffer stores recent chain samples used to construct differential evolution proposal jumps
        
        Parameters
        ----------
        iter: int
            Current iteration index
        
        burn: int
            Size of the DE buffer window
        """

        self._DEbuffer = shift_array(self._DEbuffer, -len(self._AMbuffer))  # shift DEbuffer to the left
        self._DEbuffer[-len(self._AMbuffer) :] = self._AMbuffer  # add new samples to the new empty spaces

    # SCAM jump
    def covarianceJumpProposalSCAM(self, x, iter, beta):
        """
        Single Component Adaptive Metropolis (SCAM) proposal
        
        Randomly selects a parameter group and proposes a jump using the adaptive covariance structure of that group
        
        Parameters
        ----------
        x: ndarray
            Current parameter vector
        
        iter: int
            Current iteration number
        
        beta: float
            Inverse temperature of the chain
        
        Returns
        -------
        q: ndarray
            Proposed parameter vector
        
        qxy: float
            Log Hastings ratio contribution
        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        # adjust step size
        prob = self.stream.random()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:
        #   scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on beta
        # if self.beta >= 0.01:
        #     scale *= 1/np.sqrt(self.beta)

        # get parmeters in new diagonalized basis
        # y = np.dot(self.U.T, x[self.covinds])

        # make correlated componentwise adaptive jump
        ind = np.unique(self.stream.integers(0, ndim, 1))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        q[self.groups[jumpind]] += (
            self.stream.standard_normal() * cd * np.sqrt(self.S[jumpind][ind]) * self.U[jumpind][:, ind].flatten()
        )

        return q, qxy

    # AM jump
    def covarianceJumpProposalAM(self, x, iter, beta):
        """
        Adaptive Metropolis (AM) proposal
        
        Uses the adaptive covariance estimate of the selected parameter group to generate a correlated proposal
        
        Parameters
        ----------
        x: ndarray
            Current parameter vector
        
        iter: int
            Current iteration number
        
        beta: float
            Inverse temperature of the chain
        
        Returns
        -------
        q: ndarray
            Proposed parameter vector
        
        qxy: float
            Log Hastings ratio contribution
        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))

        # adjust step size
        prob = self.stream.random()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:
        #    scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on beta
        # if self.beta >= 0.01:
        #     scale *= 1/np.sqrt(self.beta)

        # get parmeters in new diagonalized basis
        y = np.dot(self.U[jumpind].T, x[self.groups[jumpind]])

        # make correlated componentwise adaptive jump
        ind = np.arange(len(self.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] = y[ind] + self.stream.standard_normal(neff) * cd * np.sqrt(self.S[jumpind][ind])
        q[self.groups[jumpind]] = np.dot(self.U[jumpind], y)

        return q, qxy

    # Differential evolution jump
    def DEJump(self, x, iter, beta):
        """
        Differential Evolution (DE) proposal
        
        Generates a proposal using the difference between two samples drawn from the DE buffer
        
        Parameters
        ----------
        x: ndarray
            Current parameter vector
        
        iter: int
            Current iteration number
        
        beta: float
            Inverse temperature of the chain
        
        Returns
        -------
        q: ndarray
            Proposed parameter vector
        
        qxy: float
            Log Hastings ratio contribution
        """

        # get old parameters
        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        bufsize = len(self._DEbuffer)

        # draw a random integer from 0 - iter
        mm = self.stream.integers(0, bufsize)
        nn = self.stream.integers(0, bufsize)

        # make sure mm and nn are not the same iteration
        while mm == nn:
            nn = self.stream.integers(0, bufsize)

        # get jump scale size
        # prob = self.stream.random()

        # mode jump
        # if prob > 0.5:
        #     scale = 1.0

        # else:
        #     scale = self.stream.random() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / self.beta)

        scale = 1.0

        for ii in range(ndim):

            # jump size
            sigma = self._DEbuffer[mm, self.groups[jumpind][ii]] - self._DEbuffer[nn, self.groups[jumpind][ii]]

            # jump
            q[self.groups[jumpind][ii]] += scale * sigma

        return q, qxy

    # add jump proposal distribution functions
    def addProposalToCycle(self, func, weight):
        """
        Add a proposal distribution to the sampler proposal cycle

        Parameters
        ----------
        func: callable
            Proposal function to add
        
        weight: int
            Relative weight of the proposal in the proposal cycle
        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            # print('ERROR: Can not have 0 weight in proposal cycle!')
            # sys.exit()
            return

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(func)

        # add to jump dictionary and initialize file
        if func.__name__ not in self.jumpDict:
            self.jumpDict[func.__name__] = [0, 0]
            fout = open(self.outDir + "/" + func.__name__ + "_jump.txt", "w")
            fout.close()

    # add auxilary jump proposal distribution functions
    def addAuxilaryJump(self, func):
        """
        Register an auxiliary jump proposal
        
        Auxiliary jumps are applied after each standard proposal. Examples include cyclic boundary corrections or parameter transformations
        
        Parameters
        ----------
        func: callable
            Auxiliary proposal function
        """

        # set auxilary jump
        self.aux.append(func)

    # randomized proposal cycle
    def randomizeProposalCycle(self):
        """
        Randomize proposal cycle that has already been filled

        """

        # get length of full cycle
        length = len(self.propCycle)

        # get random integers
        index = np.arange(length)
        self.stream.shuffle(index)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[ind] for ind in index]

    # call proposal functions from cycle
    def _jump(self, x, iter):
        """
        Randomly select and execute a proposal from the proposal cycle. Returns the proposed state, Hastings ratio contribution, and the name of the proposal used

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        ind = self.stream.integers(0, length)
        q, qxy = self.propCycle[ind](x, iter, self.beta)

        # axuilary jump
        if len(self.aux) > 0:
            for aux in self.aux:
                q, qxy_aux = aux(x, q, iter, self.beta)
                qxy += qxy_aux

        return q, qxy, self.propCycle[ind].__name__

    # TODO: jump statistics


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
