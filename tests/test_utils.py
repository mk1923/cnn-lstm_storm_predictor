import pytest
from utility_functions.dataloader import StormDataLoader, StormDataset

TEST_DATA_DIR = 'path/to/your/test/data'


@pytest.fixture
def common_dataset_instance():
    return StormDataset(TEST_DATA_DIR)


def test_common_dataset_len(common_dataset_instance):
    assert len(common_dataset_instance) > 0


def test_common_dataset_sample(common_dataset_instance):
    sample = common_dataset_instance[0]
    assert 'storm_id' in sample
    assert 'sample_number' in sample
    assert 'relative_time' in sample
    assert 'ocean' in sample
    assert 'wind_speed' in sample
    assert 'image' in sample


def test_common_dataloader(common_dataset_instance):
    # Test if the CommonDataLoader can be created without errors
    batch_size = 2
    common_dataloader = StormDataLoader(common_dataset_instance,
                                        batch_size=batch_size)
    assert common_dataloader.batch_size == batch_size

    # Test if batches can be iterated over
    for batch in common_dataloader:
        assert 'storm_id' in batch
        assert 'sample_number' in batch
        assert 'relative_time' in batch
        assert 'ocean' in batch
        assert 'wind_speed' in batch
        assert 'image' in batch


if __name__ == '__main__':
    pytest.main()
