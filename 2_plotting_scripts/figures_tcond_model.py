import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LogNorm

plt.ioff()
"""
def tcond_model_test(file_phase_solid, df_tcond, df_vle_md,
                     height=2, width=3,
                     lrs = [12, 20, 28],
                     T_lower=0.6, T_upper=1.5,
                     rho_lower=-5e-2, rho_upper=1.1,
                     color_phase='k',
                     zorder=3, capsize=2, markersize=7.):

    df_tcond = df_tcond[df_tcond['T*'] <= T_upper].reset_index(drop=True)
    df_info = file_phase_solid.parse('info')
    df_vle = file_phase_solid.parse('vle')
    df_sle = file_phase_solid.parse('sle')
    df_sve = file_phase_solid.parse('sve')

    ########
    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    bbox_labels = dict(facecolor='white',  edgecolor='white',  boxstyle='round,pad=0.2', alpha=0.6)
    for ax, label in zip(ax_list, labels):
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.text(0.9, 0.9, label, transform=ax.transAxes, bbox=bbox_labels, zorder=zorder+1,
                horizontalalignment='center', verticalalignment='center')

    axts = [ax1, ax2, ax3]
    axbs = [ax4, ax5, ax6]
    norm_tcond =LogNorm(vmin=0.4, vmax=12)
    for axt, axb, lambda_r in zip(axts, axbs, lrs):

        axt.set_title(f' $\lambda_\mathrm{{r}} = {lambda_r}, \lambda_\mathrm{{a}} = 6$')
        axt.set_ylim([T_lower, T_upper])
        axt.set_xlim([rho_lower, rho_upper])

        # Plotting phase equilibria
        df_vle_md_lr = df_vle_md[df_vle_md['lr'] == lambda_r]

        df_info_lr = df_info[df_info['lambda_r'] == lambda_r]
        df_vle_lr = df_vle[df_vle['lambda_r'] == lambda_r]
        df_sle_lr = df_sle[df_sle['lambda_r'] == lambda_r]
        df_sve_lr = df_sve[df_sve['lambda_r'] == lambda_r]

        axt.plot(df_info_lr[['rhovad_triple', 'rholad_triple', 'rhosad_triple']].values[0] , df_info_lr[['T_triple', 'T_triple','T_triple']].values[0], color=color_phase)
        # VLE
        axt.plot(df_vle_lr['rhov_vle_model'], df_vle_lr['T_vle_model'], color=color_phase)
        axt.plot(df_vle_lr['rhol_vle_model'], df_vle_lr['T_vle_model'], color=color_phase)
        # SLE
        axt.plot(df_sle_lr['rhol_sle_model'], df_sle_lr['T_sle_model'], color=color_phase)
        axt.plot(df_sle_lr['rhos_sle_model'], df_sle_lr['T_sle_model'], color=color_phase)
        # SVE
        axt.plot(df_sve_lr['rhov_sve_model'], df_sve_lr['T_sve_model'], color=color_phase)
        axt.plot(df_sve_lr['rhos_sve_model'], df_sve_lr['T_sve_model'], color=color_phase)

        # Thermal conductivity
        df_tcond_lr = df_tcond[df_tcond['lr'] == lambda_r]
        sc = axt.scatter(df_tcond_lr['rho*'], df_tcond_lr['T*'], c=df_tcond_lr['thermal_conductivity'], s=markersize, zorder=zorder,
                         cmap='viridis', norm=norm_tcond)

        ###############################
        # Thermal conductivity at VLE #
        ###############################
        # axb.set_xlim([0.6, 1.3])
        axb.set_ylim([1., 8.])

        vle_labels = []
        for rhol, Tvle in df_vle_md_lr[['rhol*', 'T*']].values:
            rhol = np.round(rhol, 6)
            Tvle = np.round(Tvle, 2)
            is_rho = df_tcond_lr['rho*'] == rhol
            is_T = df_tcond_lr['T*'] == Tvle
            is_vle = np.logical_and(is_rho, is_T)
            if np.any(is_vle):
                vle_labels.append(df_tcond_lr.index[is_vle])
        vle_labels = np.hstack(vle_labels)
        df_tcond_lr_vle = df_tcond_lr.loc[vle_labels]

        axb.errorbar(df_tcond_lr_vle['T*'], df_tcond_lr_vle['thermal_conductivity'], yerr=df_tcond_lr_vle['thermal_conductivity_std'],
                     fmt='o', color='k', markerfacecolor='white', capsize=capsize, markersize=markersize-2)
        axb.plot(df_vle_lr['T_vle_model'], df_vle_lr['tcondl_vle_model'], color='k')
        axb.plot(df_vle_lr['T_vle_model'].values[0],  df_vle_lr['tcondl_vle_model'].values[0], color='k', marker='s')
        axb.plot(df_vle_lr['T_vle_model'].values[-1],  df_vle_lr['tcondl_vle_model'].values[-1], color='k', marker='o')

    cbt = fig.colorbar(sc, ax=axts, orientation='vertical', fraction=0.02, pad=0.02)

    ax2.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')
    cbt.ax.set_title(r'$\kappa^*$')
    ax4.set_ylabel(r'$\kappa^*$')
    ax5.set_xlabel(r'$T^*$')
    return fig
"""

def plot_tcond_model(file_phase_solid, df_tcond, df_vle_md,
                     height=2, width=3,
                     lrs = [8, 12, 16, 24, 28],
                     T_lower=0.6, T_upper=1.5,
                     rho_lower=-5e-2, rho_upper=1.1,
                     color_phase='k',
                     zorder=3, capsize=2, markersize=7.):

    df_tcond = df_tcond[df_tcond['T*'] <= T_upper].reset_index(drop=True)
    df_info = file_phase_solid.parse('info')
    df_vle = file_phase_solid.parse('vle')
    df_sle = file_phase_solid.parse('sle')
    df_sve = file_phase_solid.parse('sve')

    ########
    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 4])

    ax6 = fig.add_subplot(gs[1, 0])
    ax7 = fig.add_subplot(gs[1, 1])
    ax8 = fig.add_subplot(gs[1, 2])
    ax9 = fig.add_subplot(gs[1, 3])
    ax10 = fig.add_subplot(gs[1, 4])

    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    bbox_labels = dict(facecolor='white',  edgecolor='white',  boxstyle='round,pad=0.2', alpha=0.6)
    for ax, label in zip(ax_list, labels):
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.text(0.87, 0.9, label, transform=ax.transAxes, bbox=bbox_labels, zorder=zorder+1,
                horizontalalignment='center', verticalalignment='center')

    axts = [ax1, ax2, ax3, ax4, ax5]
    axbs = [ax6, ax7, ax8, ax9, ax10]

    for ax in axts[1:]:
        ax.tick_params(labelleft=False)
    for ax in axbs[1:]:
        ax.tick_params(labelleft=False)

    norm_tcond =LogNorm(vmin=0.4, vmax=12)
    for axt, axb, lambda_r in zip(axts, axbs, lrs):

        # axt.set_title(f' $\lambda_\mathrm{{r}} = {lambda_r}, \lambda_\mathrm{{a}} = 6$')
        axt.set_title(f' $\lambda_\mathrm{{r}} = {lambda_r}$')

        axt.set_ylim([T_lower, T_upper])
        axt.set_xlim([rho_lower, rho_upper])

        # Plotting phase equilibria
        df_vle_md_lr = df_vle_md[df_vle_md['lr'] == lambda_r]

        df_info_lr = df_info[df_info['lambda_r'] == lambda_r]
        df_vle_lr = df_vle[df_vle['lambda_r'] == lambda_r]
        df_sle_lr = df_sle[df_sle['lambda_r'] == lambda_r]
        df_sve_lr = df_sve[df_sve['lambda_r'] == lambda_r]

        axt.plot(df_info_lr[['rhovad_triple', 'rholad_triple', 'rhosad_triple']].values[0] , df_info_lr[['T_triple', 'T_triple','T_triple']].values[0], color=color_phase)
        # VLE
        axt.plot(df_vle_lr['rhov_vle_model'], df_vle_lr['T_vle_model'], color=color_phase)
        axt.plot(df_vle_lr['rhol_vle_model'], df_vle_lr['T_vle_model'], color=color_phase)
        # SLE
        axt.plot(df_sle_lr['rhol_sle_model'], df_sle_lr['T_sle_model'], color=color_phase)
        axt.plot(df_sle_lr['rhos_sle_model'], df_sle_lr['T_sle_model'], color=color_phase)
        # SVE
        axt.plot(df_sve_lr['rhov_sve_model'], df_sve_lr['T_sve_model'], color=color_phase)
        axt.plot(df_sve_lr['rhos_sve_model'], df_sve_lr['T_sve_model'], color=color_phase)

        # Thermal conductivity
        df_tcond_lr = df_tcond[df_tcond['lr'] == lambda_r]
        sc = axt.scatter(df_tcond_lr['rho*'], df_tcond_lr['T*'], c=df_tcond_lr['thermal_conductivity'], s=markersize, zorder=zorder,
                         cmap='viridis', norm=norm_tcond)
        sc.set_rasterized(True)

        ###############################
        # Thermal conductivity at VLE #
        ###############################
        # axb.set_xlim([0.6, 1.3])
        axb.set_ylim([1., 8.])

        vle_labels = []
        for rhol, Tvle in df_vle_md_lr[['rhol*', 'T*']].values:
            rhol = np.round(rhol, 6)
            Tvle = np.round(Tvle, 2)
            is_rho = df_tcond_lr['rho*'] == rhol
            is_T = df_tcond_lr['T*'] == Tvle
            is_vle = np.logical_and(is_rho, is_T)
            if np.any(is_vle):
                vle_labels.append(df_tcond_lr.index[is_vle])
        vle_labels = np.hstack(vle_labels)
        df_tcond_lr_vle = df_tcond_lr.loc[vle_labels]

        axb.errorbar(df_tcond_lr_vle['T*'], df_tcond_lr_vle['thermal_conductivity'], yerr=df_tcond_lr_vle['thermal_conductivity_std'],
                     fmt='o', color='k', markerfacecolor='white', capsize=capsize, markersize=markersize-2)
        axb.plot(df_vle_lr['T_vle_model'], df_vle_lr['tcondl_vle_model'], color='k')
        axb.plot(df_vle_lr['T_vle_model'].values[0],  df_vle_lr['tcondl_vle_model'].values[0], color='k', marker='s')
        axb.plot(df_vle_lr['T_vle_model'].values[-1],  df_vle_lr['tcondl_vle_model'].values[-1], color='k', marker='o')

    cbt = fig.colorbar(sc, ax=axts, orientation='vertical', fraction=0.02, pad=0.02)

    ax3.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')
    cbt.ax.set_title(r'$\kappa^*$')
    ax6.set_ylabel(r'$\kappa^*$')
    ax8.set_xlabel(r'$T^*$')
    return fig
