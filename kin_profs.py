'''Script to plot kinetic profiles, fetching both diagnostic results and the fits obtained by Integrated Data Analysis (IDA).
'''
import numpy as np, sys
import matplotlib.pyplot as plt
plt.ion()
import aug_sfutils as sf
from scipy.interpolate import griddata, RectBivariateSpline


class aug_profile_data(dict):
    def __init__(self, shot, t_eq, eq_diag='EQI', xcoord='rhop'):
        '''Fetch and plot radial profiles of various quantities from several
        AUG diagnostics.

        Parameters
        ----------
        shot : int
            AUG shot number
        t0 : float
            Lower bound of time window of interest.
        t1 : float
            Upper bound of time window of interest.
        eq_diag : str
            Equilibrium to fetch. NB: LP data does not use the specified equilibrium.
        xcoord : str
            Radial coordinate to use (one of 'rhop' or 'dR'=R-Rsep)

        '''
        self.shot = int(shot)
        self.t_eq = float(t_eq)
        self.eq_diag = str(eq_diag)
        self.equ = sf.EQU(shot, diag=eq_diag)
        self.xcoord = str(xcoord)

        # default to 50ms time averaging for all diagnostics
        self.t0 = float(t_eq) - 5e-2
        self.t1 = float(t_eq) + 5e-2

        # fetch separatrix locations       
        out = sf.rho2rz(self.equ, 1.0, t_in=self.t_eq, coord_in='rho_pol')
        self.Rsep_contour = out[0][0][0]
        self.Zsep_contour = out[1][0][0]

        # Rsep at the outboard midplane
        self.Rsep = np.max(self.Rsep_contour)
        
    def get_geqdsk(self):
        '''Load equilibrium in OMFIT gEQDSK format.
        '''
        from omfit_classes import omfit_eqdsk
        geqdsk =omfit_eqdsk.OMFITgeqdsk('').from_aug_sfutils(
            self.shot, self.t_eq, eq_shotfile=self.eq_diag)
        return geqdsk
    
    def RZ2rhop(self, R, Z, t0, t1):
        '''Find rhop at the given R, Z values. This is designed to check on the 
        aug_sfutils rz2rho method.

        In the present form, this function takes (R, Z) coordinates at scattered spatial points
        and gives rhop values, using the magnetic equilibrium averaged between times `t0` and `t1`.

        Parameters
        ----------
        R : 1D array (npt,)
            Major radius values [m]
        Z : 1D array (npt,)
            Vertical position values [m]
        t0, t1 : floats
            Times within which the equilibrium is averaged.
        '''
        tind = slice(*self.equ.time.searchsorted([t0,t1]))

        psi2d = np.mean(self.equ.pfm[:,:,tind], axis=2)
        psi0 = np.mean(self.equ.psi0[tind], axis=0)
        psix = np.mean(self.equ.psix[tind], axis=0)
        rhop_mesh  = np.sqrt(np.maximum(0, (psi2d - psi0)/(psix - psi0)))

        return griddata((np.tile(self.equ.Rmesh,(len(self.equ.Zmesh),1)).flatten(),
                         np.tile(self.equ.Zmesh, (len(self.equ.Rmesh),1)).T.flatten()),
                        rhop_mesh.flatten(),
                        (R,Z),
                        method='linear')

    def setup_data_plot(self):

        fig,axs = plt.subplots(1,2, figsize=(12,6), sharex=True)
        self.fig = fig
        self.ax_ne = axs[0]
        self.ax_Te = axs[1]

        self.ax_ne.set_xlabel(r'$\rho_p$' if self.xcoord=='rhop' else r'$R-R_{sep}$')
        self.ax_ne.set_ylabel(r'$n_e$ [$10^{19}$ $m^{-3}$]')
        self.ax_ne.grid(ls='--')
        
        self.ax_Te.set_xlabel(r'$\rho_p$' if self.xcoord=='rhop' else r'$R-R_{sep}$')
        self.ax_Te.set_ylabel(r'$T_e$ [$eV$]')
        self.ax_Te.grid(ls='--')


    def load_vta(self, plot=False, t0=None, t1=None):
        '''Load midplane Thomson.
        '''
        vta = sf.SFREAD(self.shot, 'vta')
        if not vta.status:
            print(f'Could not load VTA for shot={shot}')
            return

        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        t_vta_c = vta.gettimebase('Ne_c')
        t_vta_e = vta.gettimebase('Ne_e')

        # time index of interest
        tind_c = slice(*t_vta_c.searchsorted([_t0, _t1]))
        tind_e = slice(*t_vta_e.searchsorted([_t0, _t1]))
        
        _R_vta_c = vta.getobject('R_core')  # R changes over time, Z doesn
        _Z_vta_c = vta.getobject('Z_core')
        _R_vta_e = vta.getobject('R_edge')
        _Z_vta_e = vta.getobject('Z_edge')

        R_vta_c = np.tile(_R_vta_c, (len(_Z_vta_c),1)).T[tind_c].flatten()
        Z_vta_c = np.tile(_Z_vta_c, (len(_R_vta_c),1))[tind_c].flatten()

        R_vta_e = np.tile(_R_vta_e, (len(_Z_vta_e),1)).T[tind_e].flatten()
        Z_vta_e = np.tile(_Z_vta_e, (len(_R_vta_e),1))[tind_e].flatten()

        # R-Rsep
        dR_c = R_vta_c - self.Rsep
        dR_e = R_vta_e - self.Rsep
        
        rhop_c = sf.rz2rho(self.equ, R_vta_c, Z_vta_c, t_in=self.t_eq, coord_out='rho_pol')[0]
        rhop_e = sf.rz2rho(self.equ, R_vta_e, Z_vta_e, t_in=self.t_eq, coord_out='rho_pol')[0]

        ne_c = vta.getobject('Ne_c')[tind_c].flatten()
        ne_e = vta.getobject('Ne_e')[tind_e].flatten()
        ne_c_unc = vta.getobject('SigNe_c')[tind_c].flatten()
        ne_e_unc = vta.getobject('SigNe_e')[tind_e].flatten()

        Te_c = vta.getobject('Te_c')[tind_c].flatten()
        Te_e = vta.getobject('Te_e')[tind_e].flatten()
        Te_c_unc = vta.getobject('SigTe_c')[tind_c].flatten()
        Te_e_unc = vta.getobject('SigTe_e')[tind_e].flatten()

        # some useful masking
        mask_c = np.logical_and(ne_c_unc<1e20, Te_c_unc<500)
        mask_e = np.logical_and(ne_e_unc<1e20, Te_e_unc<500)
        rhop_c = rhop_c[mask_c]
        rhop_e = rhop_e[mask_e]
        dR_c = dR_c[mask_c]
        dR_e = dR_e[mask_e]
        ne_c = ne_c[mask_c]
        ne_e = ne_e[mask_e]
        Te_c = Te_c[mask_c]
        Te_e = Te_e[mask_e]
        ne_c_unc = ne_c_unc[mask_c]
        ne_e_unc = ne_e_unc[mask_e]
        Te_c_unc = Te_c_unc[mask_c]
        Te_e_unc = Te_e_unc[mask_e]
        
        if plot:
            if not hasattr(self, 'ax_ne'): self.setup_data_plot()

            xx_c = rhop_c if self.xcoord=='rhop' else dR_c
            xx_e = rhop_e if self.xcoord=='rhop' else dR_e
            
            self.ax_ne.errorbar(xx_c, ne_c/1e19, ne_c_unc/1e19, fmt='.', label='CTS')
            self.ax_ne.errorbar(xx_e, ne_e/1e19, ne_e_unc/1e19, fmt='.', label='ETS')

            self.ax_Te.errorbar(xx_c, Te_c, Te_c_unc, fmt='.', label='CTS')
            self.ax_Te.errorbar(xx_e, Te_e, Te_e_unc, fmt='.', label='ETS')

        self['vta'] = {
            'core': {'rhop': rhop_c, 'dR': dR_c, 'ne': ne_c, 'ne_unc': ne_c_unc, 'Te': Te_c, 'Te_unc': Te_c_unc},
            'edge': {'rhop': rhop_e, 'dR': dR_e, 'ne': ne_e, 'ne_unc': ne_e_unc, 'Te': Te_e, 'Te_unc': Te_e_unc}
        }

    
    def load_dtn(self, plot=False, t0=None, t1=None):
        ''' Divertor Thomson.
        If `t0` and `t1` are provided, they override those used in the 
        class initialization.
        '''
        dtn = sf.SFREAD(self.shot, 'dtn')
        if not dtn.status:
            print(f'Could not load DTN for shot={self.shot}')
            return

        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        t_dtn = dtn.gettimebase('Te_ld')
        tind = slice(*t_dtn.searchsorted([_t0,_t1]))

        ne_dtn = dtn.getobject('Ne_ld')[tind].flatten()
        ne_dtn_unc = dtn.getobject('SigNe_ld')[tind].flatten()
        Te_dtn = dtn.getobject('Te_ld')[tind].flatten()
        Te_dtn_unc = dtn.getobject('SigTe_ld')[tind].flatten()

        # R and Z are indpt of time, but we want them for every time point
        _R_dtn = dtn.getobject('R_ld')
        _Z_dtn = dtn.getobject('Z_ld')
        _rhop = sf.rz2rho(self.equ, _R_dtn, _Z_dtn, t_in=self.t_eq, coord_out='rho_pol')[0]

        rhop = np.repeat(_rhop[None], len(t_dtn[tind]), axis=0).flatten()
        R_dtn = np.repeat(_R_dtn[None], len(t_dtn[tind]), axis=0).flatten()
        Z_dtn = np.repeat(_Z_dtn[None], len(t_dtn[tind]), axis=0).flatten()
        
        mask = np.logical_or(ne_dtn_unc/1e19 >1., Te_dtn_unc > 100.)
        if plot:
            if not hasattr(self, 'ax_ne'): self.setup_data_plot()

            xx = rhop[mask] if self.xcoord=='rhop' else dR
            self.ax_ne.errorbar(rhop[mask], ne_dtn[mask]/1e19, ne_dtn_unc[mask]/1e19,
                                fmt='.', label='DTS')
            self.ax_Te.errorbar(rhop[mask], Te_dtn[mask], Te_dtn_unc[mask],
                                fmt='.', label='DTS')

        self['dtn'] = {'rhop': rhop[mask], 'R': R_dtn[mask], 'Z': Z_dtn[mask], 
                       'ne': ne_dtn[mask], 'ne_unc': ne_dtn_unc[mask],
                       'Te': Te_dtn[mask], 'Te_unc': Te_dtn_unc[mask]}
    
    def load_lin(self, plot=False, t0=None, t1=None):
        '''Load lithium beam data.
        Current version only gives ne.
        '''
        #40006
        lin = sf.SFREAD(self.shot, 'lin')
        if not lin.status:
            print(f'Could not load LIN for shot={self.shot}')
            return

        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        t_lin = lin.gettimebase('time')
        tind = slice(*t_lin.searchsorted([_t0,_t1]))
        
        ne_lin = lin.getobject('ne')[tind].flatten()
        ne_lin_unc = lin.getobject('ne_unc')[tind].flatten()
        nelow_lin = lin.getobject('ne_lo')[tind].flatten()
        neup_lin = lin.getobject('ne_up')[tind].flatten()
        R_lin = lin.getobject('R')[:,tind].flatten()
        Z_lin = lin.getobject('Z')[:,tind].flatten()

        # R-Rsep
        dR = R_lin - self.Rsep

        rhop = sf.rz2rho(self.equ, R_lin, Z_lin, t_in=self.t_eq, coord_out='rho_pol')[0]

        if plot:
            if not hasattr(self, 'ax_ne'): self.setup_data_plot()
            xx = rhop if self.xcoord=='rhop' else dR
            
            self.ax_ne.errorbar(dR, ne_lin/1e19, ne_lin_unc/1e19, fmt='.', label='LiB')

        self['lin'] = {'rhop': rhop, 'dR': dR, 'ne': ne_lin, 'ne_unc': ne_lin_unc}

    def load_cec(self, plot=False, thin=100, t0=None, t1=None):
        '''Load Electron Cyclotron Emission (ECE) data, labelled "CEC" on AUG.
        '''
        #40166
        cec = sf.SFREAD(self.shot, 'cec')
        if not cec.status:
            print(f'Could not load CEC for shot={self.shot}')
            return

        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        t_cec = cec.getobject('time-A')
        tind = slice(*t_cec.searchsorted([_t0, _t1]),thin)

        Trad_cec = cec.getobject('Trad-A')
        Trad_cec_unc = Trad_cec * 0.15 # 15% arbitrary uncertainty
        timeRZ_cec = cec.getobject('rztime')
        _R_cec = cec.getobject('R-A')
        _Z_cec = cec.getobject('z-A')

        # R,Z locations are reported at lower time res in the shotfile
        R_cec = np.zeros((len(t_cec[tind]), _R_cec.shape[1]))
        Z_cec = np.zeros((len(t_cec[tind]), _R_cec.shape[1]))
        for ch in np.arange(_R_cec.shape[1]):
            R_cec[:,ch] = np.interp(t_cec[tind], timeRZ_cec, _R_cec[:,ch])
            Z_cec[:,ch] = np.interp(t_cec[tind], timeRZ_cec, _Z_cec[:,ch])

        rhop = sf.rz2rho(self.equ, R_cec.flatten(), Z_cec.flatten(), t_in=self.t_eq, coord_out='rho_pol')[0]

        # R-Rsep
        dR = R_cec - self.Rsep
        
        if plot:
            if not hasattr(self, 'ax_Te'): self.setup_data_plot()

            xx = rhop if self.xcoord=='rhop' else dR
            self.ax_Te.errorbar(xx, Trad_cec[tind].flatten(),
                                Trad_cec_unc[tind].flatten(), fmt='.', label='ECE')

        self['cec'] = {'rhop': rhop, 'dR': dR, 'Te': Trad_cec[tind].flatten(), 'Te_unc': Trad_cec_unc[tind].flatten()} 

    def load_ida(self, plot=False, t0=None, t1=None):
        '''Load Integrated Data Analysis (IDA) fits.
        '''

        ida = sf.SFREAD(self.shot, 'ida')
        if not ida.status:
            print(f'Could not load IDA for shot={self.shot}')
            return

        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        tida = ida.gettimebase('Te')
        tind = slice(*tida.searchsorted([_t0, _t1]))
        
        rhop = ida.getareabase('Te')
        Te = ida.getobject('Te')
        Te_unc = ida.getobject('Te_unc')
        ne = ida.getobject('ne')
        ne_unc = ida.getobject('ne_unc')

        # complicated way of mapping to R-Rsep...
        out = sf.rho2rz(self.equ, np.mean(rhop[:,tind],axis=1), t_in=self.t_eq, coord_in='rho_pol')
        Rmid = []
        #Rgeo = out[0][0][0]
        for ii in np.arange(len(out[0][0])):
            if len(out[1][0][ii])>1:
                mask = out[1][0][ii]<self.equ.R0 #Rgeo
            else:
                mask = [True,]*len(out[1][0][ii])
            ind = np.argmin(np.abs(out[1][0][ii][mask]))
            Rmid.append(out[0][0][ii][mask][ind])
        dR = np.array(Rmid) - self.Rsep
        
        if plot:
            if not hasattr(self, 'ax_ne'): self.setup_data_plot()

            xx = np.mean(rhop[:,tind],axis=1) if self.xcoord=='rhop' else dR
            
            self.ax_ne.plot(xx, np.mean(ne[:,tind],axis=1)/1e19, label='IDA')
            self.ax_Te.plot(xx,  np.mean(Te[:,tind],axis=1), label='IDA')
                            
        self['ida'] = {'rhop': np.mean(rhop[:,tind],axis=1), 'dR': dR,
                       'ne': np.mean(ne[:,tind],axis=1), 'Te': np.mean(Te[:,tind],axis=1)}

            
    def load_cnz(self, plot=False, edition=1, elm_filter=False, t0=None, t1=None):
        '''Load CXRS edge data (CNZ). Requires atomID processing routines.
        '''
        # time window
        _t0 = self.t0 if t0 is None else t0
        _t1 = self.t1 if t1 is None else t1
        
        from atomID.AUG.cxrs.CXRS_profs import CXRS_prof
        shot = 31497
        inte = CXRS_prof("CNZ", shot, "inte", edition=edition, fcoord=True)
        ti = CXRS_prof("CNZ", shot, "Ti_c", edition=edition, fcoord=True)
        vrot = CXRS_prof("CNZ", shot, "vrot", edition=edition, fcoord=True)

        if elm_filter:
            ELM_experiment = os.environ.get('LOGNAME')
            ti.conditionalTime(_t0, _t1, experiment=ELM_experiment,
                               diagnostic2plot='XVS', signal2plot='S2L2A07')

        rhop, Ti, Ti_unc = ti(_t0, _t1)
        
        if plot:
            if not hasattr(self, 'ax_ne'): self.setup_data_plot()
    
            ax_Te.errorbar(rhop, Ti, Ti_unc, fmt='.', label='CNZ')

        self['cnz'] = {'rhop': rhop, 'Ti': Ti, 'Ti_unc': Ti_unc}



if __name__=='__main__':

    try:
        shot = int(sys.argv[1])
    except:
        shot = 40470

    try:
        t0 = float(sys.argv[2])
    except:
        t0 = 2.5

    try:
        t1 = float(sys.argv[3])
    except:
        t1 = 3.5

    # initialize class
    data = aug_profile_data(shot, (t0+t1)/2.)

    data.load_ida(plot=True)
    data.load_vta(plot=True)
    data.load_dtn(plot=True)
    data.load_cec(plot=True)
    data.load_lin(plot=True)
    
    data.ax_ne.legend(loc='best').set_draggable(True)
    data.ax_Te.legend(loc='best').set_draggable(True)
