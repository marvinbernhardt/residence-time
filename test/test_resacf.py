#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from resacf import resacf
import unittest


class TestResacfMethods(unittest.TestCase):

    def test_get_lengths_of_True_spans(self):
        sequence_lengths_list = [
            ([False, True, False, True, False], [1, 1], [1, 1]),
            ([False, False, True, False, True, False, False], [1, 1], [1, 1]),
            ([True, True, True, False], [], [3]),
            ([False, True, True, True], [], [3]),
            ([True, True, True], [], [3]),
        ]

        for sequence, lengths_False, lengths_True in sequence_lengths_list:
            lengths_test = resacf.get_lengths_of_True_spans(sequence, outer_spans=False)
            self.assertIsNone(
                np.testing.assert_array_equal(lengths_test, lengths_False))
            lengths_test = resacf.get_lengths_of_True_spans(sequence, outer_spans=True)
            self.assertIsNone(
                np.testing.assert_array_equal(lengths_test, lengths_True))


if __name__ == '__main__':
    unittest.main()
