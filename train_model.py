# Standard library
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from neural_lam import constants, utils
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


class DynamicARStepCallback(pl.Callback):
    """
    Progressively update number of ar steps during training
    """

    def __init__(self, train_loader, change_epochs, ar_steps, checkpoint_dir):
        super().__init__()
        self.train_loader = train_loader
        self.change_epochs = change_epochs
        self.ar_steps = ar_steps
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.change_epochs:
            new_ar_step = self.ar_steps[self.change_epochs.index(epoch)]
            self.train_loader.dataset.update_pred_length(new_ar_step)
            print(f"Epoch {epoch}: Updating AR steps to {new_ar_step}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch + 1 in self.change_epochs:
            trainer.save_checkpoint(f"{self.checkpoint_dir}/epoch_{epoch}.ckpt")


def main():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )

    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset, corresponding to name in data directory "
        "(default: mediterranean)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hi_lam",
        help="Model architecture to train/evaluate (default: hi_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1,
        help="Number of nodes to run job on (default: 1)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical",
        help="Graph to load and use in graph-based model "
        "(default: hierarchical)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-5) "
        "(default: 1)",
    )
    parser.add_argument(
        "--finetune_start",
        type=float,
        default=0.8,
        help="Fraction of epochs after which ar steps are increased "
        "(default: 0.8)",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        choices=["analysis", "reanalysis"],
        default=None,
        help="Type of data to use: 'analysis' or 'reanalysis' (default: None)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in days to consider single time step " "(default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine"],
        default=None,
        help="learning rate decay (default: None)",
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=1e-5,
        help="Initial learning rate (default: 0.00001)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs (default: 5)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    args = parser.parse_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 5, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            args.dataset,
            pred_length=1,
            split="train",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            data_subset=args.data_subset,
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    max_pred_length = (6 // args.step_length) - 2  # 4
    val_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            args.dataset,
            pred_length=max_pred_length,
            split="val",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            data_subset=args.data_subset,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )
    checkpoint_dir = f"saved_models/{run_name}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )
    change_epochs, ar_steps = utils.get_ar_steps(
        args.epochs, args.ar_steps, args.finetune_start
    )
    print("Change epochs:", change_epochs, "AR steps:", ar_steps)
    dynamic_ar_callback = DynamicARStepCallback(
        train_loader, change_epochs, ar_steps, checkpoint_dir
    )
    logger = pl.loggers.WandbLogger(
        project=constants.WANDB_PROJECT, name=run_name, config=args
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        num_nodes=args.n_nodes,
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, dynamic_ar_callback],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    args.dataset,
                    pred_length=max_pred_length,
                    split="test",
                    subsample_step=args.step_length,
                    subset=bool(args.subset_ds),
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


if __name__ == "__main__":
    main()
