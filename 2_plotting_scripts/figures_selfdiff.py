import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Patch
from scipy.interpolate import interp1d


plt.ioff()


def plot_self_diffusivity_parity(width, height,
                                 component_list,
                                 component_names,
                                 data_dict_models,
                                 other_properties_limits,
                                 markersize=4, fontsize_annotation=8):
    ########

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for ax in ax_list:
        ax.tick_params(direction='in', which='both')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel(r'$D^{\rm lit}$ / $\mathrm{m^2s^{-1}}$')
        ax.set_ylabel(r'$D^{\rm pred}$ / $\mathrm{m^2s^{-1}}$')

    model_types = ['vle', 'vle_visc', 'vle_sle_sve'] 
    title_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    dict_kwargs = dict()
    dict_kwargs['vle'] = dict(linestyle='', color='C0', marker='s', label=r'OF$_1$')
    dict_kwargs['vle_visc'] = dict(linestyle='', color='C2', marker='v', label=r'OF$_2$')
    dict_kwargs['vle_sle_sve'] = dict(linestyle='', color='C1', marker='o', label=r'OF$_3$')

    for component_name, ax, title in zip(component_list, ax_list, title_list):
        # setting title
        ax.set_title(title + ' ' + component_names[component_name])

        dict_models = data_dict_models[component_name]
        ylabels_pos = [0.85, 0.72, 0.60] 
        for model_type, ylabel_pos in zip(model_types, ylabels_pos):
            # Reading the self diffusivity data
            dict_values = dict_models['dict_values'][model_type]
            T_diff_lit = dict_values['T_diff_lit']
            rho_diff_lit = dict_values['rho_diff_lit']
            diff_lit = dict_values['diff_lit']
            diff_model = np.array(dict_values['diff_model'])
            diff_mape = dict_values['diff_mape']
            phase_diff = dict_values['phase_diff']

            # reading phase boundaries
            df_info = dict_models[f'{model_type}_solid_file'].parse('info')
            # df_vle = dict_models[f'{model_type}_solid_file'].parse('vle')
            df_sle = dict_models[f'{model_type}_solid_file'].parse('sle')

            # Checking if the model predicts a frozen point
            is_stable = np.ones_like(T_diff_lit, dtype=bool)
            # checking if temperature is above the triple point
            is_stable[T_diff_lit < df_info['T_triple_model'].values] = False

            # checking if the density is below the density of the liquid SLE
            rhol_sle_intp = interp1d(df_sle['T_sle_model'].to_numpy(), df_sle['rhol_sle_model'].to_numpy(), fill_value='extrapolate')
            is_stable[rho_diff_lit > rhol_sle_intp(T_diff_lit)] = False

            #####################
            # Plotting the data #
            #####################
            ax.plot(diff_lit[is_stable], diff_model[is_stable], **dict_kwargs[model_type], markersize=markersize)
            ax.plot(diff_lit[~is_stable], diff_model[~is_stable], **dict_kwargs[model_type], markersize=markersize, markerfacecolor='white')

            mape = 100. * np.nanmean(np.abs(diff_model[is_stable]/diff_lit[is_stable] - 1.))
            ax.text(0.05, ylabel_pos, 'MAPE ' + dict_kwargs[model_type]['label'] + f': {mape:.2f}\%',
                    transform=ax.transAxes, fontsize=fontsize_annotation, color=dict_kwargs[model_type]['color'])

        diff_lower = other_properties_limits[component_name]['diff_lower']
        diff_upper = other_properties_limits[component_name]['diff_upper']
        ax.set_xlim([diff_lower, diff_upper])
        ax.set_ylim([diff_lower, diff_upper])
        ax.plot([diff_lower, diff_upper], [diff_lower, diff_upper], color='k')

    return fig
