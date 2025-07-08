import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.config import BeachSegConfig
from src.data import BeachSegDataModule
from src.model import PromptModel
from src.util import setup_logger

logger = logging.getLogger(__name__)


def main():
    """
    trains model
    """
    base_conf = OmegaConf.structured(BeachSegConfig)
    cli_conf = OmegaConf.from_cli()
    # cli_conf = OmegaConf.load("train.yaml")
    assert isinstance(cli_conf, DictConfig)

    conf: BeachSegConfig = OmegaConf.merge(base_conf, cli_conf)  # type: ignore

    # Skip in DDP
    rank = os.environ.get("NODE_RANK", None)
    model_training_root = Path(conf.model_training_root) / conf.project / "train"
    model_training_root.mkdir(exist_ok=True, parents=True)
    runs = [int(p.name) for p in model_training_root.iterdir() if not p.name.startswith(".")]
    last_id = max(runs + [-1])
    if rank is None:
        model_dir = model_training_root / str(last_id + 1).zfill(5)

        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "wandb").mkdir(exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        log_file_name = "log.log"
    else:
        model_dir = model_training_root / str(last_id).zfill(5)
        log_file_name = f"log_{rank}.log"

    # Setup logger
    setup_logger(logger, model_dir, log_file_name)

    logger.info(f"Saving results to {model_dir}")
    logger.info(f"Training semantic segmentation model with classes {conf.classes}")

    seed_everything(conf.seed, workers=True)

    # if rank is None:
    #     num_gpus = 1  # max(1, calc_num_gpus(conf))

    #     assert (
    #         conf.world_size == num_gpus
    #     ), f"Must set config world_size equal to number of gpus. world_size: {conf.world_size}. gpus: {num_gpus}"

    # Load model and data
    datamodule = BeachSegDataModule(config=conf)
    model = PromptModel(conf, datamodule)

    # wandb_logger = WandbLogger(log_model="all", project=conf.project, save_dir=model_dir)
    tensorboard_logger = TensorBoardLogger(model_dir)
    csv_logger = CSVLogger(model_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir / "checkpoints",
        monitor=conf.monitor_metric,
        mode=conf.monitor_mode,
        every_n_epochs=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor("epoch")
    # bs_finder = BatchSizeFinder()
    # device_stats = DeviceStatsMonitor()

    assert len(conf.devices) > 0

    devices = list(map(int, conf.devices)) if len(conf.devices) > 1 else conf.devices[0]
    trainer = Trainer(
        max_epochs=conf.epochs,
        default_root_dir=model_dir / "checkpoints",
        precision=conf.precision,  # type: ignore
        logger=[tensorboard_logger, csv_logger],  # wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],  # bs_finder, device_stats
        deterministic=conf.deterministic,
        devices=devices,
        accelerator=conf.accelerator,
        log_every_n_steps=conf.log_every_n_steps,
        accumulate_grad_batches=13,
    )

    if trainer.is_global_zero:
        # Save conf
        OmegaConf.save(config=conf, f=model_dir / "conf.yaml")
        # wandb_logger.log_hyperparams(dict(conf))  # type: ignore

    # Train!
    trainer.fit(model=model, datamodule=datamodule)  # type: ignore

    if trainer.is_global_zero:
        with open(model_dir / "classes.txt", "w") as f:
            f.write("\n".join(conf.classes))

    logger.info("Done!")


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
