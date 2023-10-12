from mne.io import RawArray
from mne import create_info


def plot_imf(imf, sfreq=256, name='fig', save=False):
    length = imf.shape[0]
    ch_names = [str(i) for i in range(length)]
    info = create_info(ch_names, sfreq, verbose=False)
    if save:
        RawArray(imf, info).plot(scalings='auto').savefig(name, dpi=999)
    else:
        RawArray(imf, info).plot(scalings='auto')


