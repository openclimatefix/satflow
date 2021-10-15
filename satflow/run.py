"""Command line entrypoint to train a satflow model from a config file"""
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    """
    Train a satflow model
    
    https://hydra.cc/docs/intro/

    Args:
        config: the configuration values will be provided by hydra based
            on how the script is executed from the command line

    Returns: the output of model training
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from satflow.core import utils
    from satflow.experiments.train import train

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    #

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
