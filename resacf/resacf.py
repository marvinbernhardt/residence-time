#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Can be used as a script to open a gmx select (Gromacs) output file
   and create the autocorrelation function from it. This is often used
   to calculate the residence time.
"""

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Calculate residence time from selection file')
    parser.add_argument('infile', type=str,
                        help=('a file containing selections for several frames as '
                              + 'generated by gmx select -oi'))
    parser.add_argument('outfile', type=str,
                        help='Outfile containing t, acf, acf_std')
    parser.add_argument('-e', '--end', type=float,
                        help='time of last frame to consider')
    parser.add_argument('-m', '--max-false', type=float, default=0.0,
                        help='maximum time length of non-selection to be ignored.'
                             ' Default: 0.0')
    parser.add_argument('-n', '--n-blocks', type=int, default=5,
                        help='number of blocks to be used for block averaging.'
                             ' Default: 5')
    parser.add_argument('-o', '--outer-spans', action='store_const',
                        const=True, default=False,
                        help='consider spans that are at begin or end of file')
    parser.add_argument('-d', '--delay', action='store_const',
                        const=True, default=False,
                        help='delay all true spans by the -m time instead of filling '
                             'small false blocks')
    # parse arguments
    args = parser.parse_args()
    # calculate acf
    t, acf, acf_std = calc_acf_from_select_data(args.infile,
                                                end_time=args.end,
                                                max_False_time=args.max_false,
                                                n_blocks=args.n_blocks,
                                                outer_spans=args.outer_spans,
                                                delay=args.delay)
    # save results
    np.savetxt(args.outfile, np.vstack((t, acf, acf_std)).T)


def get_lengths_of_True_spans(bool_array, outer_spans=False):
    assert len(bool_array) > 0
    # indices + 1 because we take diff() to be difference to preceding element
    indices = np.argwhere(np.diff(bool_array, n=1)).flatten() + 1
    if outer_spans:
        indices = np.hstack(([0], np.array(indices), [len(bool_array)]))
    # ofset: start at nth span border
    if len(indices >= 2):
        offset = int(not bool_array[indices[0]])
        span_lengths = np.diff(indices)[offset::2]
    else:
        span_lengths = np.array([])
    return span_lengths


def remove_small_False_spans(bool_array, max_False_span):
    # jumps in the bool_array
    jumps = np.argwhere(np.diff(bool_array, n=1)).flatten() + 1
    # indices on jumps pointing to small spans
    small_spans = (np.argwhere(np.diff(jumps) <= max_False_span)).flatten()
    # iterate small spans
    for small_span in np.vstack((jumps[small_spans], jumps[small_spans+1])).T:
        # if small False span
        if not bool_array[small_span[0]]:
            # make it True
            bool_array[small_span[0]:small_span[1]] = True


def delay_true_spans(bool_array, delay):
    len_array = len(bool_array)
    # jumps in the bool_array
    jumps = np.argwhere(np.diff(bool_array, n=1)).flatten() + 1
    jumps = np.hstack((jumps, len_array))
    # iterate small spans
    for span in np.vstack((jumps[:-1], jumps[1:])).T:
        # if False span
        if not bool_array[span[0]]:
            # add delay
            bool_array[span[0]:min(span[0]+delay, len_array)] = True


def add_acf_from_span_lengths(t, acf, span_lengths):
    for span_len in span_lengths:
        # add picewise linear
        acf[:span_len] += (span_len - t[:span_len])


def calc_acf_from_select_data(filename, max_False_time=0.0, n_blocks=5,
                              outer_spans=False, end_time=None, int_type='UInt16',
                              delay=False):
    df = pd.read_csv(filename, sep=' ', header=None, skipinitialspace=True,
                     usecols=[0, 1])
    n_atoms_max = df[1].max()
    dt = (df.iat[-1, 0] - df.iat[0, 0]) / (len(df[0]) - 1)
    n_rows = int(end_time / dt)
    max_False_span = int(max_False_time / dt)
    df = pd.read_csv(filename, sep=' ', header=None, skipinitialspace=True,
                     names=range(n_atoms_max + 2), nrows=n_rows)
    del df[0]
    del df[1]
    # pandas UInt16 allows nan, up to 65,535
    # needs to be changed for systems with more atoms
    df = df.astype(int_type)
    # integer time(step)
    t = np.array(df.index)
    acf_blocks = np.zeros((n_blocks, len(t)), dtype='float')
    print('doing blocks', end=' ')
    for block in range(n_blocks):
        print(block, end=' ', flush=True)
        acf = np.zeros_like(t, dtype='int64')
        norm = 0
        # every n_block atom for block averaging
        for atom in pd.unique(df.values.ravel('K'))[block::n_blocks]:
            # selection status for every frame
            selected = np.array((df == atom).any(axis=1), dtype=np.bool)
            # remove small False spans
            if max_False_span > 0:
                if delay:
                    delay_true_spans(selected, max_False_span)
                else:
                    remove_small_False_spans(selected, max_False_span)
            # lengths of continuous selection
            span_lengths = get_lengths_of_True_spans(selected, outer_spans=outer_spans)
            norm += np.sum(span_lengths)
            # calculate
            add_acf_from_span_lengths(t, acf, span_lengths)
        # normalize acf
        acf = acf.astype(np.float) / norm
        acf_blocks[block, :] = acf
    print('finished blocks')
    # block averaging
    acf_mean = acf_blocks.mean(axis=0)
    acf_std = acf_blocks.std(axis=0)
    # real time
    t = t.astype(np.float) * dt
    return t, acf_mean, acf_std


if __name__ == "__main__":
    main()
