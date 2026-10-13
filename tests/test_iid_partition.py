"""IID allocation invariants and composition checks (no training required)."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from src.config import constants
from src.data_ops.iid_partition import partition_iid_indices


FLAGS = ('IID_RANDOM', 'IID_STRATIFIED', 'IID_BALANCED', 'IID_UNEQUAL')


class IIDPartitionTests(unittest.TestCase):
    def test_all_combinations_are_disjoint_reproducible_and_preserve_pool(self):
        labels = np.repeat([2, 7, 12], [120, 75, 45])
        for enabled in itertools.product([False, True], repeat=4):
            if not any(enabled):
                continue
            with self.subTest(enabled=enabled), patch.multiple(constants, **dict(zip(FLAGS, enabled))):
                parts = partition_iid_indices(labels, 5, seed=42)
                again = partition_iid_indices(labels, 5, seed=42)
                indices = np.concatenate(parts)
                self.assertEqual(len(indices), len(np.unique(indices)))
                self.assertTrue(all(len(part) >= 2 for part in parts))
                for first, second in zip(parts, again):
                    np.testing.assert_array_equal(first, second)
                expected = [45, 45, 45] if enabled[2] else [120, 75, 45]
                np.testing.assert_array_equal(np.unique(labels[indices], return_counts=True)[1], expected)
                sizes = [len(part) for part in parts]
                if enabled[3]:
                    self.assertGreater(max(sizes) - min(sizes), 1)
                else:
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                if enabled[1] or (enabled[2] and not enabled[0]):
                    proportions = np.array(expected) / sum(expected)
                    for part in parts:
                        actual = np.array([(labels[part] == c).sum() for c in [2, 7, 12]])
                        self.assertTrue(np.all(np.abs(actual - len(part) * proportions) <= 1.01))

    def test_small_classes_do_not_create_empty_clients(self):
        with patch.multiple(constants, IID_RANDOM=False, IID_STRATIFIED=True,
                            IID_BALANCED=False, IID_UNEQUAL=False):
            parts = partition_iid_indices(np.arange(30), 15)
            self.assertEqual([len(part) for part in parts], [2] * 15)

    def test_invalid_settings_and_insufficient_balanced_data(self):
        with patch.multiple(constants, **dict.fromkeys(FLAGS, False)):
            with self.assertRaises(ValueError):
                partition_iid_indices([0, 1], 1)
        with patch.object(constants, 'IID_RANDOM', 1):
            with self.assertRaises(ValueError):
                constants.get_iid_techniques()
        with patch.multiple(constants, IID_RANDOM=False, IID_STRATIFIED=False,
                            IID_BALANCED=True, IID_UNEQUAL=False):
            with self.assertRaisesRegex(ValueError, 'Balanced'):
                partition_iid_indices([0] * 20 + [1], 3)
        for alpha in [0, -1, np.nan, np.inf]:
            with patch.multiple(constants, IID_UNEQUAL=True, IID_UNEQUAL_ALPHA=alpha):
                with self.assertRaisesRegex(ValueError, 'ALPHA'):
                    partition_iid_indices(np.arange(20), 3)


if __name__ == '__main__':
    unittest.main()
