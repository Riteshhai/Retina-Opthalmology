import pandas as pd
import pytest

from qsart.training.data import read_manifest


def test_duplicate_connected_groups_cannot_cross_partitions_when_manifest_is_edited(tmp_path):
    path = tmp_path/'software_manifest.csv'
    pd.DataFrame({'image_path': ['fixture_a.png', 'fixture_b.png'],
                  'split': ['train', 'test'], 'split_group_id': ['same_component', 'same_component']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='split_group_id'):
        read_manifest(path)
