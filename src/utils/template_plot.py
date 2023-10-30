import numpy as np
import matplotlib.pyplot as plt
from utils.kinematics import particle_att_utils
import os
from scipy import stats


def ratio_plot(compare_hist,
               hist_value,
               ratio_hist_value,
               bins,
               ks_test_value,
               xlabel,
               save_path,
               save_name,
               ratio_hist_err,
               wd_test_value=None,
               title=None):
    fig, [ax0, ax1] = plt.subplots(2,
                                   gridspec_kw={
                                       'height_ratios': (3.5, 1),
                                       'hspace': 0.
                                   },
                                   figsize=(4, 4))
    bin_half_width = (bins[1] - bins[0]) / 2
    bin_middle_value = bin_half_width + bins[:-1]
    ax0.plot(bin_middle_value, compare_hist, label='MC(S Sample)', color='blue')
    if wd_test_value is None:
        ax0_label = 'SWD \n$P_{KS}$' + '={:.0f}%'.format(ks_test_value * 100)
    else:
        ax0_label = 'SWD \n$P_{KS}$' + '={:.0f}%'.format(
            ks_test_value * 100) + ', $P_{WD}$' + '={:.0f}%'.format(wd_test_value * 100)
    ax0.errorbar(bin_middle_value,
                 hist_value,
                 xerr=bin_half_width,
                 color='black',
                 fmt='o',
                 ms=2,
                 label=ax0_label)
    ax0.set_xlim([bins.min(), bins.max()])
    ax0.set_ylabel('Number Of Events(Normalized)')
    ylim_min = np.min((ax0.get_ylim()[0], 0))
    ylim_max = ax0.get_ylim()[1]
    ax0.legend(loc='upper left', fontsize=6)
    ax0.text(bin_middle_value[int(3 / 5 * bin_middle_value.size)],
             1.1 * ylim_max,
             '$\psi(2S) \\rightarrow \phi \pi^+ \pi^-$' +
             '\n$\phi \\rightarrow K^+ K^-$',
             fontsize=6)
    ax0.set_ylim([ylim_min, 1.3 * ylim_max])
    ax0.set_xticks([])
    if title is not None:
        ax0.set_title(title, fontsize=6)
    ax1.errorbar(bin_middle_value,
                 ratio_hist_value,
                 xerr=bin_half_width,
                 yerr=ratio_hist_err,
                 color='black',
                 ms=2,
                 fmt='o')
    ax1.axhline(y=1., linestyle='--', color='blue', lw=0.5)
    ax1.set_xlim([bins.min(), bins.max()])
    ax1.set_ylim([0.8, 1.19])
    ax1.set_xlabel(xlabel, horizontalalignment='right', x=1.0, fontsize=6)
    ax1.set_ylabel('Ratio To\nMC', fontsize=6)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, save_name) + '.jpg',
                bbox_inches='tight',
                dpi=300)
    fig.savefig(os.path.join(save_path, save_name) + '.pdf',
                bbox_inches='tight',
                dpi=300)
    plt.close(fig)


def get_mass_square(momentum):
    return momentum[:, 3] - np.sum(momentum[:, :3], axis=1)


def plot_line(values, ylabel, save_path, save_name, values_err=None):
    fig, ax = plt.subplots()
    ax.plot(values, color='black')
    if values_err is not None:
        ax.plot(values + values_err, alpha=0.1, color='black', lw=0.1)
        ax.plot(values - values_err, alpha=0.1, color='black', lw=0.1)
        ax.fill_between(np.arange(values.shape[0]),
                        values - values_err,
                        values + values_err,
                        alpha=0.1,
                        color='black')
    ax.set_yscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, save_name) + '.jpg',
                bbox_inches='tight',
                dpi=300)
    fig.savefig(os.path.join(save_path, save_name) + '.pdf',
                bbox_inches='tight',
                dpi=300)
    plt.close(fig)
