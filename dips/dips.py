"""
dips: detrending periodic signals
"""

import sys
import multiprocessing as mp

import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from scipy.stats import binned_statistic as hstats

class Dips(object):
    """
    The main Dips class that exists so that the dips namespace is clean on both POSIX
    and non-POSIX systems.
    """


    def __init__(self, finput=None, 
                    data=None,
                    bins=200,
                    origin=0.0,
                    period=1.0,
                    logfile=None,
                    tol=1e-8,
                    diff=2e-5,
                    step_size=1e-3,
                    attenuation_factor=0.9,
                    allow_upstep=False,
                    cols=[0,1],
                    disable_mp=False,
                    initial_pdf='median',
                    interim_prefix=None,
                    jitter=0.0,
                    normalize_data=False,
                    output_prefix=None,
                    renormalize=False,
                    save_interim=0,
                    yonly=False):

        '''
        Initialize a Dips class with all the arguments required for computation.

        Parameters
        ----------
        finput: str,
            input file containing time, flux and optional flux error
        data: array_like,
            data in format [times, fluxes, sigmas (optional)]
        bins: int,
            number of synchronous pdf bins
        origin: float,
            the zero-point of the time-series
        period: float,
            period of the synchronous signal
        logfile: str,
            log file to send output to instead of screen
        tol: float,
            tolerance for convergence
        diff: float,
            finite difference size
        step-size: float,
            initial down-step multiplier
        attenuation_factor: float,
            attenuation factor for xi
        allow-upstep: bool,
            allow step size to increase during convergence
        cols: int
            a list of input columns to be parsed, starting from 0
        disable-mp    action='store_true
            disable multiprocessing (force serial computation)
        initial-pdf: str,
            choice of pdf initialization [\'flat\\'mean\\'median\\'random\or external filename]
        interim-prefixtype=str,
            filename prefix for interim results
        jitter: float,
            add jitter to the computed gradients
        normalize-data: bool
            normalize input data by median-dividing the fluxes
        output-prefix: str,
            filename prefix for saving results
        renormalize: bool
            force pdf normalization to 1 after every iteration
        save-interim: int,
            save interim solutions every N iterations
        yonly: bool
            use only y-distance instead of full euclidian distance
        '''

        # collect all args
        # TODO: clean up unnecessary ones
        self.bins=bins
        self.origin=origin
        self.period=period
        self.logfile=logfile
        self.tolerance=tol
        self.difference=diff
        self.xi=step_size
        self.attenuation=attenuation_factor
        self.allow_upstep=allow_upstep
        self.cols=cols
        self.disable_mp=disable_mp
        self.initial_pdf=initial_pdf
        self.interim_prefix=interim_prefix
        self.jitter=jitter
        self.normalize_data=normalize_data
        self.output_prefix=output_prefix
        self.renormalize=renormalize
        self.save_interim=save_interim
        self.yonly=yonly

        # TODO: the following block is commented out for further polish.
        # if mode is unfold, then we only need to do that and we're done.
        # if args['mode'] == 'unfold':
        #     self.data = np.loadtxt(self.finput, usecols=self.cols)
        #     self.pdf = np.loadtxt(self.initial_pdf, usecols=(1,))
        #     self.ranges = np.linspace(0, 1, self.bins+1)
        #     self.phases = self.fold(self.data[:,0], self.origin, self.period)
        #     fluxes = self.unfold(self.pdf) + np.random.normal(0.0, args['stdev'], len(self.data[:,0]))
        #     np.savetxt('test', np.vstack((self.data[:,0], fluxes, np.ones(len(fluxes))*args['stdev'])).T)
        #     exit()


        # can't serialize a file stream for multiprocessing, so don't do self.log.
        if self.logfile is not None:
            log = open(self.logfile, 'w')
        else:
            log = sys.stdout

        log.write('# Issued command:\n')
        log.write('#   %s\n# \n' % ' '.join(sys.argv))

        if data is not None:
            self.data = data
            log.write('# input data: %d rows, %d columns\n' % (self.data.shape[0], self.data.shape[1]))
        elif finput is not None:
            self.finput = finput
            self.data = np.loadtxt(finput, usecols=self.cols)
            log.write('# input data: %d rows, %d columns read in from %s\n' % (self.data.shape[0], self.data.shape[1], self.finput))
        else:
            raise ValueError("Both data and finput cannot be None. Please provide one.")
        
        if self.normalize_data:
            self.data[:,1] /= np.median(self.data[:,1])

        self.ranges = np.linspace(0, 1, self.bins+1)
        self.bin_centers = (self.ranges[1:]+self.ranges[:-1])/2
        self.phases = self.fold(self.data[:,0], self.origin, self.period)

        log.write('# initial pdf source: %s\n' % self.initial_pdf)
        if self.initial_pdf == 'flat':
            self.pdf = np.ones(self.bins)
        elif self.initial_pdf == 'mean':
            self.pdf = hstats(x=self.phases, values=self.data[:,1], statistic='mean', bins=self.bins, range=(0, 1))[0]
        elif self.initial_pdf == 'median':
            self.pdf = hstats(x=self.phases, values=self.data[:,1], statistic='median', bins=self.bins, range=(0, 1))[0]
        elif self.initial_pdf == 'random':
            means = hstats(x=self.phases, values=self.data[:,1], statistic='mean', bins=self.bins, range=(0, 1))[0]
            stds = hstats(x=self.phases, values=self.data[:,1], statistic='std', bins=self.bins, range=(0, 1))[0]
            self.pdf = np.random.normal(means, stds)
        else:
            self.pdf = np.loadtxt(self.initial_pdf, usecols=(1,))
            if len(self.pdf) != self.bins:
                log.write('#   rebinning the input pdf from %d to %d\n' % (len(self.pdf), self.bins))
                old_ranges = np.linspace(0, 1, len(self.pdf)+1)
                self.pdf = np.interp(self.bin_centers, (old_ranges[1:]+old_ranges[:-1])/2, self.pdf)

        log.write('# number of requested pdf bins: %d\n' % self.bins)

        nelems_per_bin, _ = np.histogram(self.phases, bins=self.bins)
        log.write('# number of observations per bin:\n')
        log.write('#   min: %d   max: %d   mean: %d\n# \n' % (nelems_per_bin.min(), nelems_per_bin.max(), nelems_per_bin.mean()))

        nprocs = 1 if self.disable_mp else mp.cpu_count()
        log.write('# dips running on %d %s (multiprocessing %s)\n# \n' % (nprocs, 'core' if nprocs == 1 else 'cores', 'off' if self.disable_mp else 'on'))

        # mpl.rcParams['font.size'] = 24
        # plt.figure(figsize=(16,6))
        # plt.ylim(0.9, 1.01)
        # plt.xlabel('Phase')
        # plt.ylabel('Normalized flux')
        # plt.plot(self.phases, self.data[:,1], 'b.')
        # plt.bar(0.5*(self.ranges[:-1]+self.ranges[1:]), self.pdf, width=1./self.bins, color='yellow', edgecolor='black', zorder=10, alpha=0.4)
        # plt.show()

        resids = self.data[:,1] - self.unfold(self.pdf)
        log.write('# original timeseries length:  %f\n' % self.length(self.data[:,0], self.data[:,1]))
        log.write('# initial asynchronous length: %f\n# \n' % self.length(self.data[:,0], resids))

        log.write('# computational parameters:\n')
        log.write('#   tolerance (tol):  %6.2e\n' % self.tolerance)
        log.write('#   difference (dxk): %6.2e\n' % self.difference)
        log.write('#   step size (xi):   %6.2e\n' % self.xi)
        log.write('#   attenuation (af): %6.2e\n' % self.attenuation)
        log.write('#   up-step allowed:  %s\n'    % self.allow_upstep)
        log.write('#   yonly:            %s\n'    % self.yonly)
        log.write('#   slope jitter:     %2.2f\n' % self.jitter)
        log.write('#   renormalize:      %s\n'    % self.renormalize)
        log.write('# \n')

        if self.logfile is not None:
            log.close()

    def run(self):
        # can't serialize a file stream for multiprocessing, so don't do self.log.
        if self.logfile is not None:
            log = open(self.logfile, 'a')
        else:
            log = sys.stdout

        if self.save_interim > 0:
            interim_prefix = self.finput if self.interim_prefix is None else self.interim_prefix

        log.write('# %3s %14s %12s %14s %14s %14s %14s\n' % ('it', 'async_length', 'sync_length', 'difference', 'step_size', 'mean_slope', 'merit'))

        # starting iterations
        i = 0
        l1 = self.length(self.data[:,0], self.data[:,1] - self.unfold(self.pdf))
        if self.disable_mp:
            slopes = np.array([self.slope(k) for k in range(self.bins)])
        else:
            with mp.Pool() as pool:
                slopes = np.array(pool.map(self.slope, range(self.bins)))
        mean_slope = np.abs(slopes).mean()

        while self.xi*mean_slope > self.tolerance:
            l0 = l1
            safety_counter = 0
            while True:
                steps = -self.xi*slopes
                l1 = self.length(self.data[:,0], self.data[:,1] - self.unfold(self.pdf+steps))
                if l1 > l0 and safety_counter < 1000:
                    self.xi *= self.attenuation
                    safety_counter += 1
                else:
                    self.pdf += steps
                    if self.renormalize:
                        self.pdf /= self.pdf.mean()
                    if self.allow_upstep:
                        self.xi /= self.attenuation
                    break

            if self.save_interim > 0 and i % self.save_interim == 0:
                np.savetxt('%s.%05d.ranges' % (interim_prefix, i), self.ranges)
                np.savetxt('%s.%05d.signal' % (interim_prefix, i), np.vstack((0.5*(self.ranges[:-1]+self.ranges[1:]), self.pdf)).T)
                np.savetxt('%s.%05d.trend'  % (interim_prefix, i), np.vstack((self.data[:,0], self.data[:,1]-self.unfold(self.pdf))).T)

            i += 1
            log.write('%5d %14.8f %12.8f %14.8e %14.8e %14.8e %14.8e\n' % (i, l1, self.synclength(self.pdf), l0-l1, self.xi, mean_slope, self.xi*mean_slope))

            if self.disable_mp:
                slopes = np.array([self.slope(k) for k in range(self.bins)])
            else:
                with mp.Pool() as pool:
                    slopes = np.array(pool.map(self.slope, range(self.bins)))

            mean_slope = np.abs(slopes).mean()

        prefix = self.finput if self.output_prefix is None else self.output_prefix

        np.savetxt('%s.ranges' % (prefix), self.ranges)
        np.savetxt('%s.signal' % (prefix), np.vstack((0.5*(self.ranges[:-1]+self.ranges[1:]), self.pdf)).T)
        np.savetxt('%s.trend' % (prefix), np.vstack((self.data[:,0], self.data[:,1]-self.unfold(self.pdf))).T)

        if self.logfile is not None:
            log.close()

    def fold(self, t, t0, P):
        return ((t-t0) % P) / P

    def unfold(self, pdf):
        return interp(self.bin_centers, pdf, k=3)(self.phases)

    def length(self, t, y):
        if self.yonly:
            return np.abs(y[1:]-y[:-1]).sum()
        else:
            return (((y[1:]-y[:-1])**2 + (t[1:]-t[:-1])**2)**0.5).sum()

    def synclength(self, pdf):
        if self.yonly:
            return np.abs(pdf[1:]-pdf[:-1]).sum()
        else:
            return (((pdf[1:]-pdf[:-1])**2 + (1./len(pdf))**2)**0.5).sum()

    def slope(self, k):
        x = self.pdf.copy()
        x[k] -= self.difference/2
        l1 = self.length(self.data[:,0], self.data[:,1] - self.unfold(x))
        x[k] += self.difference
        l2 = self.length(self.data[:,0], self.data[:,1] - self.unfold(x))
        if self.jitter == 0:
            return (l2-l1)/self.difference
        else:
            return np.random.normal((l2-l1)/self.difference, self.jitter*np.abs((l2-l1)/self.difference))
