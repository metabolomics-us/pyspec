import matplotlib.pyplot as plt
from typing import List

from pyspec.loader import Spectra


def _normalize(s, scale: float = 100):
    """
    internal function to normalize a spectrum to the given scale
    :param s:
    :param scale: maximum intensity
    :return:
    """

    if type(s) == Spectra:
        s = s.spectra
    if type(s) == str:
        s = [tuple(map(float, x.split(':'))) for x in s.split()]

    if scale is None or not scale:
        return s
    else:
        max_intensity = max(x[1] for x in s)
        return sorted([(x[0], scale * x[1] / max_intensity) for x in s], key=lambda x: -x[1])


def _plot_spectrum(spectrum, ax, reverse=False, scale=100, show_labels=True, n_labels=8, label_min_intensity=1,
                   label_font_size=6, mz_min=None, mz_max=None):
    """
    internal function to plot a spectrum on an axis
    :param spectrum:
    :param ax: matplotlib axis to plot onto
    :param reverse: whether this is a reference spectrum (bottom half of a head-to-tail plot)
    :param scale: maximum intensity (or False/None for no scaling)
    :param show_labels: whether to display ion labels
    :param n_labels: maximum number of ion labels to display
    :param label_min_intensity: minimum relative intensity for ion labels to be shown
    :param label_font_size: font size for ion labels
    :return:
    """

    labels = []

    for i, (mz, intensity) in enumerate(_normalize(spectrum, scale=scale)):
        ax.plot([mz, mz], [0, intensity], 'r-' if reverse else 'b-', linewidth=1.25)

        if mz_min or mz_max:
            mz_min = mz_min if mz_min else ax.get_xlim()[0]
            mz_max = mz_max if mz_max else ax.get_xlim()[1]
            ax.set_xlim((mz_min, mz_max))

        # Plot ion labels
        if show_labels and len(labels) < n_labels and intensity > label_min_intensity and all(
                abs(mz - x) > 10 for x in labels):
            height = intensity + 2
            valign = 'top' if reverse else 'bottom'

            ax.text(mz, height, '%0.4f' % mz, fontsize=label_font_size, horizontalalignment='center',
                    verticalalignment=valign)
            labels.append(mz)


def plot_mass_spectrum(spectrum, filename: str = None, title: str = None, mz_min=None, mz_max=None):
    """
    plot a single mass spectrum
    :param spectrum:
    :param filename:
    :param title:
    :return:
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_spectrum(spectrum, ax, mz_min=mz_min, mz_max=mz_max)

    if title is not None:
        plt.title(title)

    plt.ylim(0, 110)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


def _multiplot_config(fig, title: str = None):
    """
    general tasks to perform for multiplots
    :param fig:
    :param title:
    :return:
    """

    # set title
    if title is not None:
        plt.suptitle(title)

    # remove spacing between subplots
    fig.subplots_adjust(hspace=0)

    # only ticks on last subplot should be visible
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # axis labels using an additional full subplot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')


def plot_multiple_mass_spectra(spectra: List, filename: str, scale=100, labels: List[str] = None, title: str = None,
                               mz_min=None, mz_max=None):
    """
    plots a set of mass spectra in a single plot
    :param spectra:
    :param filename:
    :param scale:
    :param labels: labels for each subplot
    :param title:
    :return:
    """

    fig_height = max(6, 1.5 * len(spectra))
    fig, axes = plt.subplots(len(spectra), sharex=True, figsize=(12, fig_height))

    if labels is None:
        for s, ax in zip(spectra, axes):
            _plot_spectrum(s, ax, scale=scale, mz_min=mz_min, mz_max=mz_max)
            ax.set_ylim(0, 110)

    else:
        assert (len(labels) == len(spectra))

        for s, ax, label in zip(spectra, axes, labels):
            _plot_spectrum(s, ax, scale=scale, mz_min=mz_min, mz_max=mz_max)
            ax.set_ylim(0, 110)
            ax.set_title(label, y=0.9, fontsize=8, verticalalignment='top')

    _multiplot_config(fig, title=title)

    plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_head_to_tail_mass_spectra(spectrumA, spectrumB, filename: str, title: str = None, mz_min=None, mz_max=None):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 5))

    _plot_spectrum(spectrumA, ax1, mz_min=mz_min, mz_max=mz_max)
    _plot_spectrum(spectrumB, ax2, mz_min=mz_min, mz_max=mz_max, reverse=True)

    ax1.set_ylim(0, 110)
    ax2.set_ylim(0, 110)
    ax2.invert_yaxis()

    _multiplot_config(fig, title=title)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
