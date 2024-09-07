from t3d.tools.profile_plot import plotter
from t3d.Profiles import GridProfile

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import sys

'''
TQ - 9/4/2024
'''

try:
    fname = sys.argv[1]
except:
    fname = "out-2h/wgi-06h.log.npy"
print(f"reading {fname}")

plotter=plotter()
plotter.read_data(fname)

# set defaults
plotter.tidx = np.arange(len(plotter.time))
plotter.show_newton_iterations = False
plotter.ridx = slice(None)

#plotter.plot_panel()

save = False
outdir='plots'

plot_state_profile = False
plot_power_profile = False
temp_heat_flux = True
plot_relu = False
plot_time_dynamic = False

# set up colors
self=plotter
self.warm_map = pylab.cm.autumn(np.linspace(1, 0.25, self.N))
self.cool_map = pylab.cm.Blues(np.linspace(0.25, 1, self.N))
self.green_map = pylab.cm.YlGn(np.linspace(0.25, 1, self.N))

stop_id = np.argwhere(np.diff(self.time[::-1])<0)[0,0] + 1

if plot_state_profile:
    fig,axs = plt.subplots(1,3, figsize=(8,3.5))
    ax1, ax2, ax3 = axs

    ''' Function to plot the temperature profiles '''

    for i in self.tidx:
        t_curr = self.time[i]
        if i > 0 and not self.show_newton_iterations:
            if t_curr == self.time[i - 1]:
                # skip repeated time from Newton iteration
                continue
        # plot profiles
        tag = f'$t =${t_curr:.2f} s'
        if i == 0:
            ax1.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i], label=tag)
            ax2.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i], label=tag)
        elif i == self.N - stop_id:
            ax1.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i], label=tag)
            ax2.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i], label=tag)

            # single density plot
            ax3.plot(self.rho, self.n[i], '.-', color=self.green_map[i])
        else:
            ax1.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i])
            ax2.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i])

    ax1.set_title(r'$T_i$ [keV]')
    ax2.set_title(r'$T_e$ [keV]')
    ax3.set_title(r'$N_e$ [$10^{20}$ / $m^3$]')
    for a in axs:
        a.set_ylim(bottom=0)
        a.set_xlim(left=0.0)
        a.set_xlabel(self.rad_label)
        a.grid(self.grid_lines)
    for a in [ax1,ax2]:
        leg = a.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)

    fig.tight_layout()

    if save:
        name = "1-state-profile"
        plt.savefig(f"{outdir}/{name}.png")
        plt.savefig(f"{outdir}/{name}.svg")
        plt.savefig(f"{outdir}/{name}.pdf")



if plot_power_profile:
    fig,axs = plt.subplots(1,5, figsize=(14,3.5), sharex=True)

    ''' Function to plot the temperature profiles '''

    # calc div Q
    norm = self.data['norms']
    to_MW = norm['P_ref_MWm3'] * norm['a_minor']**3
    geo = self.data['geo_obj']
    G = geo.grho_grid/geo.area_grid
    drho = self.data['grid']['drho']


    for i in self.tidx:
        t_curr = self.time[i]
    
        specs = self.data['species'][i]
        D = specs.ref_species
        e = specs['electron']
        Qi = D.qflux
        Qe = e.qflux
        Qgb = D.p().plus()**2.5/D.n().plus()**1.5 / geo.Btor**2
        Qgbm = D.p().minus()**2.5/D.n().minus()**1.5 / geo.Btor**2
        Fpi = Qi * Qgb * to_MW * geo.area
        Fmi = Qi.minus() * Qgbm * to_MW * geo.area.minus()
        Fpe = Qe * Qgb * to_MW * geo.area
        Fme = Qe.minus() * Qgbm * to_MW * geo.area.minus()
        divQi = -G[:-1]*(Fpi-Fmi)/drho
        divQe = -G[:-1]*(Fpe-Fme)/drho

        if i > 0 and not self.show_newton_iterations:
            if t_curr == self.time[i - 1]:
                # skip repeated time from Newton iteration
                continue
        # plot profiles
        tag = f'$t =${t_curr:.2f} s'
        if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue

        if i == self.N - stop_id:

            # fusion power
            axs[0].plot(self.rho, self.Sp_alpha_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
            axs[0].plot(self.rho, self.Sp_alpha_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')
            axs[0].plot(self.rho, self.Palpha_MWm3[i], '.-', color=self.green_map[i], label='$S_{p,{tot}}$')

            # radiation
            axs[1].plot(self.rho, self.Sp_rad_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
            axs[1].plot(self.rho, self.Sp_rad_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')

            # turbulence
            axs[2].plot(self.rho[:-1], divQi, '.-', color=self.warm_map[i], fillstyle='none', label=r'$-\nabla \cdot Q_i$')
            axs[2].plot(self.rho[:-1], divQe, '.-', color=self.cool_map[i], fillstyle='none', label=r'$-\nabla \cdot Q_e$')

            # collisional exchange
            axs[3].plot(self.rho, self.Sp_coll_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label=r'$S^\mathrm{coll}_{p,i}$')
            axs[3].plot(self.rho, self.Sp_coll_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label=r'$S^\mathrm{coll}_{p,e}$')

        else:
            axs[0].plot(self.rho, self.Sp_alpha_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
            axs[0].plot(self.rho, self.Sp_alpha_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])
            axs[0].plot(self.rho, self.Palpha_MWm3[i], '.-', color=self.green_map[i])

            axs[1].plot(self.rho, self.Sp_rad_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
            axs[1].plot(self.rho, self.Sp_rad_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])
            axs[2].plot(self.rho[:-1], divQi, '.-', color=self.warm_map[i], fillstyle='none')
            axs[2].plot(self.rho[:-1], divQe, '.-', color=self.cool_map[i], fillstyle='none')
            #axs[2].plot(self.midpoints, self.Q_MW_i['tot'][i], 'o-', color=self.warm_map[i], fillstyle='none')
            #axs[2].plot(self.midpoints, self.Q_MW_e['tot'][i], 'o-', color=self.cool_map[i], fillstyle='none')

            axs[3].plot(self.rho, self.Sp_coll_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
            axs[3].plot(self.rho, self.Sp_coll_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])

    # aux power
    i = self.tidx[-1]
    axs[-1].plot(self.rho, self.Sp_aux_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
    axs[-1].plot(self.rho, self.Sp_aux_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')

    _, top = axs[-1].get_ylim()
    axs[-1].set_ylim(bottom=0, top=1.5 * top)

    # Q sink
    Qi_tot = self.Q_MW_i['tot'][i,-1]
    Qe_tot = self.Q_MW_e['tot'][i,-1]
    #Qi_tot = np.sum(self.Q_MW_i['tot'][i])
    #Qe_tot = np.sum(self.Q_MW_e['tot'][i])
#    axs[2].plot(self.midpoints, self.Q_MW_i['tot'][i], 'o-', color=self.warm_map[i], fillstyle='none', label=f'$Q_i$')
#    axs[2].plot(self.midpoints, self.Q_MW_e['tot'][i], 'o-', color=self.cool_map[i], fillstyle='none', label=f'$Q_e$')

    axs[0].set_title('fusion power density \n [MW/m$^{3}$]')
    axs[1].set_title('radiation [MW/m$^{3}$]')
    axs[2].set_title('turbulent heat flux \n [MW/m$^3$]')
    axs[3].set_title('collisional exchange \n [MW/m$^{3}$]')
    axs[-1].set_title('auxiliary power source \n [MW/m$^{3}$]')


    leg = axs[3].legend(loc='best', title=r'$P^{coll}_i = $' + f'{self.Sp_coll_int_MW_i[-1]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.65)
    
    leg = axs[2].legend(loc='best', title='$Q_{turb} =$' + f'{-(Qi_tot + Qe_tot):.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.65)

    title = (r'$P_{\alpha} = $' + f'{self.Palpha_int_MW[i]:.2f} MW\n' +
             r'$P_{fus} = $' + f'{5.03 * self.Palpha_int_MW[i]:.2f} MW')
    leg = axs[0].legend(loc='best', title=title, fancybox=False, shadow=False, ncol=1, fontsize=8)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.65)

    leg = axs[1].legend(loc='best', title=r'$P_{rad} = $' + f'{self.Prad_tot[-1]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.65)

    leg = axs[-1].legend(loc='best', title='$P_{aux} =$' + f'{self.Sp_aux_int_MW_tot[i]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.65)

    for a in axs:
        a.set_xlim(left=0.0)
        a.set_xlabel(self.rad_label)
        a.grid(self.grid_lines)

    fig.tight_layout()

    if save:
        name = "2-power-profile"
        plt.savefig(f"{outdir}/{name}.png")
        plt.savefig(f"{outdir}/{name}.svg")
        plt.savefig(f"{outdir}/{name}.pdf")


if temp_heat_flux:
    fig,axs = plt.subplots(6,1, figsize=(3.5,9), sharex=True)
    self.grid_lines = True

    ''' Function to plot the temperature profiles '''

    stop_id = np.argwhere(np.diff(self.time[::-1])<0)[0,0] + 1
    i = self.tidx[-1]


    # fusion power
    axs[0].plot(self.rho, self.Ti[i] , 'o-', color=self.warm_map[i], label='$T_i [keV]$')

    # radiation
    axs[1].plot(self.midpoints,self.aLTi[i], 'x-', color=self.warm_map[i], label='$a/L_{T_i}$')

    D = self.data['species'][i].ref_species
    #Qi = self.Qi['tot'][i]
    Qi = D.qflux
    axs[2].plot(self.midpoints,Qi, 'x-', color=self.warm_map[i], label='$Q_i$ [GB]')

    geo = self.data['geo_obj']
    Qgb = D.p().plus()**2.5/D.n().plus()**1.5 / geo.Btor**2
    Qgbm = D.p().minus()**2.5/D.n().minus()**1.5 / geo.Btor**2
    axs[3].plot(self.midpoints,Qgb, 'x-', color=self.warm_map[i], label='$Q_{gb}$ [keV n20]')

    axs[4].plot(self.midpoints,Qi * Qgb, 'x-', color=self.warm_map[i], label='$Q_i$ [keV n20]')


    drho = self.data['grid']['drho']
    G = geo.grho_grid/geo.area_grid
    Fp = Qi * Qgb * geo.area
    Fm = Qi.minus() * Qgbm * geo.area.minus()
    tag =r'$ -\nabla \cdot Q_i$ [dp/dt]'
    axs[5].plot(self.rho[:-1],-G[:-1]*(Fp-Fm)/drho, 'o-', color=self.warm_map[i], label=tag)
    
    axs[-1].set_xlabel(self.rad_label)

    for a in axs:
        a.set_xlim(left=0.0)
        a.grid(self.grid_lines)
        a.legend()

    fig.tight_layout()

    # second panel
    fig,axs = plt.subplots(6,1, figsize=(4,8), sharex=True)
    norm = self.data['norms']
    to_MW = norm['P_ref_MWm3'] * norm['a_minor']**3
    axs[0].plot(self.midpoints,Qi * Qgb * to_MW, 'x-', color=self.warm_map[i], label='$Q_i$ [MW/m$^2$]')
    axs[1].plot(self.rho, geo.area_grid, 'o-', color=self.green_map[i], label='A [$m^2$]')
    axs[1].plot(self.midpoints, geo.area, 'x', color=self.warm_map[i], label='A [$m^2$]')
    axs[2].plot(self.rho, geo.grho_grid, 'o-', color=self.green_map[i], label=r'$\nabla \rho$ [$m^2$]')
    #axs[2].plot(self.midpoints, geo.grho, 'x-', color=self.green_map[i], label=r'$\nabla \rho$ [$m^2$]')
    axs[3].plot(self.rho, G, 'o-', color=self.green_map[i], label=r"$G = 1/V' = \nabla\rho/A$ [$m^{-3}$]")
    Fp = Qi * Qgb * to_MW * geo.area
    Fm = Qi.minus() * Qgbm * to_MW * geo.area.minus()
    axs[4].plot(self.midpoints,Fp, 'x-', color=self.warm_map[i], label=r'$ Q_i \cdot A = F_+$ [MW]')
    axs[4].plot(self.midpoints,Fm, 'x-', color=self.cool_map[i], label=r'$F_-$')
    tag =r'$ \nabla \cdot Q_i = G(F_+ - F_-)/\Delta \rho$ [MW/m$^3$]'
    axs[5].plot(self.rho[:-1],G[:-1]*(Fp-Fm)/drho, 'o-', color=self.warm_map[i], label=tag)
    for a in axs:
        a.set_xlim(left=0.0)
        a.grid(self.grid_lines)
        a.legend()
    axs[-1].set_xlabel(self.rad_label)

    fig.tight_layout()

if plot_relu:
    fig,axs = plt.subplots(1,1, figsize=(5,5))
    ''' Function to plot q '''
    r = self.ridx
    for i in self.tidx:
        t_curr = self.time[i]
        if i > 0 and not self.show_newton_iterations:
            if t_curr == self.time[i - 1]:
                # skip repeated time from Newton iteration
                continue
        axs.plot(self.aLpi[i][r] - self.aLn[i][r], self.Qi['tot'][i][r], '.', color=self.warm_map[i])
        axs.plot(self.aLpe[i][r] - self.aLn[i][r], self.Qe['tot'][i][r], '.', color=self.cool_map[i])
    axs.set_title(r'$Q(L_T)$ [GB]')
    axs.set_xlabel('$a/L_T$')
    axs.grid()


if plot_time_dynamic:
    fig,axs = plt.subplots(1,5, figsize=(14,3.5), sharex=True)

    data = self.data
    t_time = np.array(data['t'])
    t_step = data['t_step_idx']  # time step (ignoring repeats)
    i_max = t_step[-1]
    # get the index before each change
    args = [t_step.index(i+1) - 1 for i in np.arange(i_max)]

    P_aux_t = self.Sp_aux_int_MW_tot

    total_pressure = (self.pe + self.pi) * 1e20 * 1.609e-16 / 1e6
    convert = self.geo.volume_integrate
    grid = self.geo.grid
    W_tot_t = np.array([convert(GridProfile(p,grid)) for p in total_pressure])  # MJ

    tauE_t = W_tot_t / P_aux_t

    P_alpha_t = np.array(self.Palpha_int_MW)
    Q_power_t = 5.03*P_alpha_t/P_aux_t

    n0 = self.n[:,0] * 1e20
    navg = np.mean(self.n,axis=-1) * 1e20
    nedge = self.n[:,-2] * 1e20

    Ti0 = self.Ti[:,0]  # keV
    nTtau_t = n0*Ti0*tauE_t
    ntau_t = n0*tauE_t

    t_rms = self.data['t_rms']
    # calculate sudo density limit
    P_ext_total = P_alpha_t + P_aux_t
    # checked that (R,B) includes scaling
    B0 = np.mean(self.geo.Btor)  # avg over r-axis
    R0 = self.geo.a_minor * self.geo.AspectRatio
    a0 = self.geo.a_minor

    V0 = (np.pi * a0 * a0) * (2*np.pi*R0)
    n_sudo_t = np.sqrt(P_ext_total * B0 / V0)

    P_rad_t = np.array(self.Prad_tot)

    # calculate beta
    p_avg = W_tot_t*1e6 / V0
    mu0 = 4*np.pi/1e7
    beta_t = p_avg / B0**2 * (2 * mu0)

    t_SI = self.time
    #axs[0].plot(t_SI, W_tot_t, '--')
    axs[0].plot(t_SI[args], W_tot_t[args], 'C1',label="Wtot [MJ]")
    ax2 = axs[0].twinx()
    ax2.plot(t_SI[args], 100*beta_t[args], 'C2', label="Beta [%]")
    ax2.legend(loc=4)

    #axs[1].plot(t_SI, Q_power_t)
    axs[1].plot(t_SI[args], Q_power_t[args], 'C1', label='Q')
    axs[1].axhline(1, ls='--', color='r', label='breakeven')

    axs[2].plot(t_SI[args], nTtau_t[args],'C4', label=r'$n_0 T_{i0} \tau_E [s \cdot keV / m^3]$')
    axs[2].axhline(3e21, ls='--', color='r', label='lawson')
    axs[2].set_yscale('log')

    axs[3].plot(t_SI, P_aux_t, label=r'$P_\mathrm{aux}$ [MW]')
    axs[3].plot(t_SI, P_alpha_t, label=r'$P_\alpha$ [MW]')

    axs[4].plot(t_SI, n_sudo_t, label=r'$n_{sudo}$')
    axs[4].plot(t_SI, navg/1e20, label=r'$n_{avg}$')  
    axs[4].plot(t_SI, nedge/1e20, label=r'$n_{edge}$')  
    axs[4].plot(t_SI, 2*n_sudo_t,'C0--', label=r'$2 n_{sudo}$')
    for a in axs:
        a.legend()
        a.set_xlabel("time [s]")

    fig.tight_layout()

plt.show()
