"""
Import this script as a module 
"""

import numpy as np 
from itertools import chain

def det(R, lmin=None, hist=None, verb=True):
    """returns DETERMINISM for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating DET...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    DET = num.sum() / den.sum()
    return DET

def diagonal_lines_hist(R, verb=True):
    """returns the histogram P(l) of diagonal lines of length l."""
    if verb:
        print("diagonal lines histogram...")
    line_lengths = []
    for i in range(1, len(R)):
        d = np.diag(R, k=i)
        ll = _count_num_lines(d)
        line_lengths.append(ll)
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths

def _count_num_lines(arr):
    """returns a list of line lengths contained in given array."""
    line_lens = []
    counting = False
    l = 0
    for i in range(len(arr)):
        if counting:
            if arr[i] == 0:
                l += 1
                line_lens.append(l)
                l = 0
                counting = False
            elif arr[i] == 1:
                l += 1
                if i == len(arr) - 1:
                    l += 1
                    line_lens.append(l)
        elif not counting:
            if arr[i] == 1:
                counting = True
    return line_lens

def entr(R, lmin=None, hist=None, verb=True):
    """returns ENTROPY for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating ENTR...")
    pl = nlines.astype('float') / float(len(ll))
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx1 = l >= lmin
    pl = pl[idx1]
    idx = pl > 0.
    ENTR = (-pl[idx] * np.log(pl[idx])).sum()
    return ENTR


def tau_recurrence(R):
	N=R.shape[0] # R is the Recurrence matrix
	q=np.zeros(N)
	for tau in range(N):
		q[tau]=np.diag(R,k=tau).mean()
	return q


def cpr(Rx, Ry):
    """
    Returns the correlation of probabilities of recurrence (CPR).
    """
    assert Rx.shape == Ry.shape, "RPs are of different sizes!"
    N = Rx.shape[0]
    qx, qy = [np.zeros(N) for i in range(2)]
    for tau in range(N):
        qx[tau] = np.diag(Rx, k=tau).mean()
        qy[tau] = np.diag(Ry, k=tau).mean()

    # obtain indices after taking into account decorrelation time
    e = np.exp(1.)
    try:
        ix = np.where(qx < 1. / e)[0][0]
        iy = np.where(qy < 1. / e)[0][0]
        i = max(ix, iy)
    except IndexError:
        i = N

    # final estimate
    if i < N:
        # normalised data series to mean zero and standard deviation one after
        # removing entries before decorrelation time
        qx_ = qx[i:]
        qx_ = (qx_ - np.nanmean(qx_)) / np.nanstd(qx_)
        qy_ = qy[i:]
        qy_ = (qy_ - np.nanmean(qy_)) / np.nanstd(qy_)

        # estimate CPR as the dot product of normalised series
        C = (qx_ * qy_).mean()
    else:
        C = np.nan

    return C


