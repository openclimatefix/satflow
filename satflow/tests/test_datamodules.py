from satflow.data.datamodules import SatFlowDataModule


def test_datamodule_subsetting():
    dataset = SatFlowDataModule(fake_data=True)
    dataset.setup()
    train_dset = dataset.train_dataloader()
    sample, target = next(iter(train_dset))
    assert sample["sat_data"].shape == (32, 7, 16, 16, 12)
    assert target["sat_data"].shape == (32, 48, 16, 16, 12)
