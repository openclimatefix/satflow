from satflow.data.datamodules import SatFlowDataModule


def test_datamodule_subsetting():
    dataset = SatFlowDataModule(fake_data=True)
    dataset.setup()
    train_dset = dataset.train_dataloader()
    sample = next(iter(train_dset))
    assert sample["sat_data"].size() == (32, 19, 16, 16, 12)
