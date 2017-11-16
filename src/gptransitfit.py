import numpy as np
import astropy
import pandas as pd
import matplotlib.pyplot as plt
import pyde
import pyde.de
import IPython
import ipywidgets
from IPython.display import display
from IPython.display import HTML
import mcFunc
import emcee
from collections import OrderedDict
from astropy import constants as aconst


class GPTransitFit(object):
    """
    A class that does transit fitting.
        
    NOTES:
    - Needs to have LPFunction defined
    TODO:
    """
    latex_labels_arr            = np.array([["$T_{0}$ $(\mathrm{BJD_{TDB}})$","Transit Midpoint"],
                                            ["$P$ (days)","Orbital period"],
                                            ["$R_p/R_*$","Radius ratio"],
                                            ["$R_p (R_\oplus)$","Planet radius"],
                                            ["$R_p (R_J)$","Planet radius"],
                                            ["$\delta$","Transit depth"],
                                            ["$a/R_*$","Normalized orbital radius"],
                                            ["$a$ (AU)","Semi-major axis"],
                                            ["$i$ $(^{\circ})$","Transit inclination"],
                                            ["$b$","Impact parameter"],
                                            ["$e$","Eccentricity"],
                                            ["$\omega$ $(^{\circ})$","Argument of periastron"],
                                            ["$T_{\mathrm{eq}}$(K)","Equilibrium temperature"],
                                            ["$T_{14}$ (days)","Transit duration"],
                                            ["$\\tau$ (days)","Ingress/egress duration"],
                                            ["$T_{S}$ $(\mathrm{BJD_{TDB}})$","Time of secondary eclipse"]])
    latex_labels = latex_labels_arr[:,0]
    latex_description = latex_labels_arr[:,1]
    latex_jump_labels = [r"$T_0 (BJD_{\mathrm{TBD}})$",r"$\log(P)$",r"$\cos(i)$","$R_p/R_*$","$\log(a/R_*)$","Baseline",r"\log(A)$",r"$\log(B)$"]
                                            
    def __init__(self,GPLPFunction):
        self.lpf = GPLPFunction()

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=500,mcmc=True,maximize=True):
        """
        Minimize using the PyDE
        
        NOTES:
        https://github.com/hpparvi/PyDE
        """
        centers = self.lpf.gp.get_parameter_vector()
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        print("LogLn value:",self.min_pv_lnval)
        print("Log priors",self.lpf.ps.c_log_prior(self.min_pv))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop,self.lpf.ps.ndim,self.lpf)
            pb = ipywidgets.IntProgress(max=mc_iter/50)
            display(pb)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                if i%50 == 0:
                    pb.value += 1
            print("Finished MCMC")

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps.labels,self.lpf.ps.centers,self.lpf.ps.bounds[:,0],self.lpf.ps.bounds[:,1],pv,self.lpf.ps.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def gelman_rubin(self,chains=None,burn=0,thin=1):
        """
        Calculates the gelman rubin statistic.
        
        # NOTE:
        Should be close to 1
        """
        if chains==None:
            chains = self.sampler.chain[:,burn::thin,:]
            grarray = mcFunc.gelman_rubin(chains)
        return grarray

    def plot_chains(self,labels=None,burn=0,thin=1):
        print("Plotting chains")
        if labels==None:
            labels = self.lpf.ps.descriptions
        mcFunc.plot_chains(self.sampler.chain,labels=labels,burn=burn,thin=thin)

    def plot_corner(self,labels=None,burn=0,thin=1,title_fmt='.5f',**kwargs):
        if labels==None:
            labels = self.lpf.ps.descriptions
        self.fig = mcFunc.plot_corner(self.sampler.chain,labels=labels,burn=burn,thin=thin,title_fmt=title_fmt,**kwargs)
    
    def plot_lc(self,pv=None,draw_posteriors=True,fill_between=True,OFFSET=0.98):
        if pv is None:
            pv = self.min_pv
        fig, ax = plt.subplots()
        baseline_index = self.lpf.priorDict.keys().index("baseline")
        baseline = pv[baseline_index]
        ax.plot(self.lpf.x,self.lpf.y/baseline,"k.")
        #ax.errorbar(self.lpf.x,self.lpf.y/baseline,self.lpf.yerr)
        
        if fill_between:
            self.lpf.gp.set_parameter_vector(pv)
            mu, self.cov = self.lpf.gp.predict(self.lpf.y,self.lpf.x)
            mu, self.var = self.lpf.gp.predict(self.lpf.y,self.lpf.x,return_var=True)
            mu /= baseline
            pred_std = np.sqrt(self.var)
            ax.plot(self.lpf.x,mu, color="red", alpha=1.,lw=0.5,label="GP + Transit Model")
            ax.plot(self.lpf.x,self.lpf.gp.mean.get_value(self.lpf.x)/baseline,label="Transit Model",alpha=0.5)
            ax.plot(self.lpf.x,self.lpf.y/baseline-mu+OFFSET,color="firebrick",alpha=0.8,label="Residuals",lw=0,marker="o",markersize=3)
            ax.fill_between(self.lpf.x, mu+pred_std, mu-pred_std, color="k", alpha=0.3,
                            edgecolor="none")
        
        if draw_posteriors:
            xx = np.linspace(self.lpf.x.min(), self.lpf.x.max(), 500)
            samples = self.sampler.flatchain#[:,800:,:].reshape((-1,gg.sampler.chain.shape[2]))
            for s in samples[np.random.randint(len(samples), size=12)]:
                self.lpf.gp.set_parameter_vector(s)
                baseline = s[baseline_index]
                mu = self.lpf.gp.sample_conditional(self.lpf.y, xx)/baseline
                ax.plot(xx, mu, color="#4682b4", alpha=0.3)

        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
        
        ax.set_xlabel("Date (BJD)")
        ax.set_ylabel("Relative Flux")
        ax.legend(loc="upper left",fontsize=12)

    #   def plot_nicer()
    def plot_lc_nicer(self,pv=None,draw_posteriors=True,fill_between=True,OFFSET=0.98):
        if pv is None:
            pv = self.min_pv
        fig, ax = plt.subplots()
        baseline_index = self.lpf.priorDict.keys().index("baseline")
        baseline = pv[baseline_index]
        ax.plot(self.lpf.x,self.lpf.y/baseline,"k.")
        #ax.errorbar(self.lpf.x,self.lpf.y/baseline,self.lpf.yerr)
        
        if fill_between:
            self.lpf.gp.set_parameter_vector(pv)
            mu, self.cov = self.lpf.gp.predict(self.lpf.y,self.lpf.x)
            mu, self.var = self.lpf.gp.predict(self.lpf.y,self.lpf.x,return_var=True)
            mu /= baseline
            pred_std = np.sqrt(self.var)
            ax.plot(self.lpf.x,mu, color="red", alpha=1.,lw=0.5,label="GP + Transit Model")
            ax.plot(self.lpf.x,self.lpf.gp.mean.get_value(self.lpf.x)/baseline,label="Transit Model",alpha=0.5)
            ax.plot(self.lpf.x,self.lpf.y/baseline-mu+OFFSET,color="firebrick",alpha=0.8,label="Residuals",lw=0,marker="o",markersize=3)
            ax.fill_between(self.lpf.x, mu+pred_std, mu-pred_std, color="k", alpha=0.3,
                            edgecolor="none")
        
        if draw_posteriors:
            xx = np.linspace(self.lpf.x.min(), self.lpf.x.max(), 500)
            samples = self.sampler.flatchain#[:,800:,:].reshape((-1,gg.sampler.chain.shape[2]))
            for s in samples[np.random.randint(len(samples), size=12)]:
                self.lpf.gp.set_parameter_vector(s)
                baseline = s[baseline_index]
                mu = self.lpf.gp.sample_conditional(self.lpf.y, xx)/baseline
                ax.plot(xx, mu, color="#4682b4", alpha=0.3)
    
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
        
        ax.set_xlabel("Date (BJD)",labelpad=10,fontsize=16)
        ax.set_ylabel("Relative Flux",labelpad=10,fontsize=16)
        ax.set_ylim(0.97,1.02)
        ax.set_yticks([0.97,0.98,0.99,1.00])
        ax.legend(loc="upper left",fontsize=12)
        plt.tight_layout()
        plt.savefig("../figures/light_curve_fit.pdf",bbox_inches='tight')



    def get_transit_parameters(self,flatchain=None,burn=0,thin=1,st_rad=1.0,st_raderr1=0.022,st_teff=5650.,st_teff_err1=75.,e="fixed"):
        if flatchain==None:
            flatchain = self.sampler.chain[:,burn::thin,:].reshape((-1, self.lpf.ps.ndim))
        
        print("Assuming")
        print("R_s:",st_rad,"+-",st_raderr1)
        print("Teff:",st_teff,"+-",st_teff_err1)
        # Working with the posteriors
        #sampler = self.sampler
        t0 = flatchain[:,0]
        self._p_pl_orbper = 10.**flatchain[:,1]
        cosi = flatchain[:,2]
        inc = np.arccos(flatchain[:,2])
        incdeg = inc*180./np.pi
        sini = np.sin(inc)
        RpRs = flatchain[:,3]
        depth = RpRs**2.
        aRs = 10.**flatchain[:,4]
        if e=="fixed":
            e = np.zeros(len(flatchain)) #sampler.flatchain[:,5]**2. + sampler.flatchain[:,6]**2.
            w = np.zeros(len(flatchain)) #numpy.arctan(sampler.flatchain[:,5]/sampler.flatchain[:,6])
            #w = numpy.nan_to_num(w) # removes the nans
            #w = w*180./np.pi
            ecosw = np.sqrt(e)#*flatchain[:,5]
            esinw = np.sqrt(e)#*flatchain[:,6]
        else:
            e = flatchain[:,5]**2. + flatchain[:,6]**2.
            w = np.arctan(flatchain[:,5]/flatchain[:,6])
            w = np.nan_to_num(w) # removes the nans
            w = w*180./np.pi
            ecosw = e*np.cos(w)#flatchain[:,5]
            esinw = e*np.sin(w)#flatchain[:,6]
        b = aRs*cosi*(1.-e**2.)/(1.+esinw)
        
        t14 = np.copy(b) * 0.0
        t23 = np.copy(b) * 0.0
        notgrazing = np.where(((1.+RpRs)**2. - b**2.)>0.0)
        t14[notgrazing] = self._p_pl_orbper[notgrazing]/np.pi*np.arcsin(np.sqrt((1.+RpRs[notgrazing])**2. - b[notgrazing]**2.)/(sini[notgrazing]*aRs[notgrazing]))*np.sqrt(1.-e[notgrazing]**2.)/(1.+esinw[notgrazing])
        notgrazing = np.where(((1.-RpRs)**2. - b**2.)>0.0)
        t23[notgrazing] = self._p_pl_orbper[notgrazing]/np.pi*np.arcsin(np.sqrt((1.-RpRs[notgrazing])**2. - b[notgrazing]**2.)/(sini[notgrazing]*aRs[notgrazing]))*np.sqrt(1.-e[notgrazing]**2.)/(1.+esinw[notgrazing])
        t14 = np.nan_to_num(t14) # removes the nans
        t23 = np.nan_to_num(t23) # removes the nans
        tau = (t14-t23)/2.
        Tfwhm = t14-tau
        
        # Calculate the peri. time based on tC
        nu = np.pi/2. - w # true anomaly for transit away from peri.
        E = 2.*np.arctan(np.sqrt((1.0-e)/(1.+e))*np.tan(nu/2.)) # ecc. anomaly
        M = E - e*np.sin(E) # mean anomaly
        periPhase = M/(2.*np.pi)
        tP = t0 - self._p_pl_orbper*periPhase # peri time
        
        nu = 3.*np.pi/2. - w # true anomaly for eclipse away from peri.
        E = 2.*np.arctan(np.sqrt((1.0-e)/(1.+e))*np.tan(nu/2.)) # ecc. anomaly
        M = E - e*np.sin(E) # mean anomaly
        periPhase = M/(2.*np.pi)
        tS = tP + self._p_pl_orbper*periPhase + self._p_pl_orbper
        otherside = np.where(periPhase>0.0) # correct for aliases
        tS[otherside] = tP[otherside] + self._p_pl_orbper[otherside]*periPhase[otherside]
        
        # Create sampling of stellar parameters to calculate errors on compounding derived parameters
        # Teq
        Teff = np.random.normal(loc=st_teff,scale=st_teff_err1,size=len(flatchain))
        Teq = Teff*np.sqrt(1./(2.*aRs))
        # a-au
        R_s = np.random.normal(loc=st_rad,scale=st_raderr1,size=len(flatchain)) # radius of star
        aAU = aRs * R_s * aconst.R_sun.value / aconst.au.value
        # r_e
        R_e = RpRs * R_s * aconst.R_sun.value / aconst.R_earth.value
        R_j = RpRs * R_s * aconst.R_sun.value / aconst.R_jup.value
        
        self.df_post = pd.DataFrame(zip(t0,self._p_pl_orbper,RpRs,R_e,R_j,depth,aRs,aAU,incdeg,b,e,w,Teq,t14,tau,tS),columns=["t0","per","RpRs","R_e","R_j","depth","aRs","aAU","inc","b","e","w","Teq","t14","tau","tS"])
                                    
        # ugh
        nump = tS.size
        self._p_pl_orbper = self._p_pl_orbper.reshape((nump,1))
        tS = tS.reshape((nump,1))
        depth = depth.reshape((nump,1))
        aRs = aRs.reshape((nump,1))
        incdeg = incdeg.reshape((nump,1))
        b = b.reshape((nump,1))
        Tfwhm = Tfwhm.reshape((nump,1))
        tau = tau.reshape((nump,1))
        t14 = t14.reshape((nump,1))
        ecosw = ecosw.reshape((nump,1))
        esinw = esinw.reshape((nump,1))
        e = e.reshape((nump,1))
        w = w.reshape((nump,1))
                                    
        df = mcFunc.calc_medvals2(self.df_post)
        print(len(df))
        df["values"] =  [mcFunc.latex_mean_low_up(df.medvals[i],df.minus[i],df.plus[i]) for i in range(len(df))]
        df["Labels"] =  self.latex_labels
        df["Description"] = self.latex_description
        self.df_medvals = df
        return self.df_medvals

