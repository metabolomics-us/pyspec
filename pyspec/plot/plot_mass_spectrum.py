import matplotlib.pyplot as plt


def _normalize(s, scale=100):
    """Internal function to normalize a spectrum"""

    if type(s) == str:
        s = [tuple(map(float, x.split(':'))) for x in s.split()]

    max_intensity = max(x[1] for x in s)
    return sorted([(x[0], scale * x[1] / max_intensity) for x in s], key=lambda x: -x[1])

def _plot_spectrum(spectrum, ax, reverse=False, scale=100, labels=True, n_labels=8, label_min_intensity=1, label_font_size=6):
    """Internal function to plot a spectrum on an axis"""

    labels = []
    
    for i, (mz, intensity) in enumerate(_normalize(spectrum)):
        ax.plot([mz, mz], [0, intensity], 'r-' if reverse else 'b-', linewidth=1.25)
    
        # Plot ion labels
        if len(labels) < n_labels and intensity > label_min_intensity and all(abs(mz - x) > 10 for x in labels):
            height = intensity + 2
            valign = 'top' if reverse else 'bottom'
            ax.text(mz, height, '%0.4f' % mz, fontsize=label_font_size, horizontalalignment='center', verticalalignment=valign)
            labels.append(mz)


def plot_mass_spectrum(spectrum, filename, title=None):
    fig, ax = plt.subplots()

    _plot_spectrum(spectrum, ax)

    if title is not None:
        plt.title(title)

    plt.ylim(0, 110)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')

    plt.savefig(filename, dpi=300)


def plot_head_to_tail_mass_spectra(spectrumA, spectrumB, filename, title=None):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 5))

    _plot_spectrum(spectrumA, ax1)
    _plot_spectrum(spectrumB, ax2, reverse=True)

    ax1.set_ylim(0, 110)
    ax2.set_ylim(0, 110)
    ax2.invert_yaxis()

    if title is not None:
        plt.suptitle(title)

    # Axis labels
    fig.text(0.5, 0.04, "m/z", ha="center", va="center")
    fig.text(0.05, 0.5, "Relative Intensity", ha="center", va="center", rotation=90)

    # Reomve spacing between subplots
    fig.subplots_adjust(hspace=0)

    # Remove 0 tick on secondary plot
    plt.setp(ax2.get_yticklabels()[0], visible=False)

    # Only ticks on last subplot should be visible
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    plt.savefig(filename, dpi=300)
