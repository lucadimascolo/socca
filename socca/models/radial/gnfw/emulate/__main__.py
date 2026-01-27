"""Command-line interface for training the gNFW emulator.

This module provides a CLI for training the gNFW emulator model.
It can be invoked via the console script or as a module:

.. code-block:: bash

    # Via console script (after pip install)
    gnfw-emulator --n_train 100000 --n_epochs 500

    # Via python module
    python -m socca.models.radial.gnfw.emulate --resume
"""

import argparse

from . import rerun, DEFAULT_MODEL_PATH


def main():
    """Run the gNFW emulator training from the command line."""
    parser = argparse.ArgumentParser(
        description="Train or resume training of a gNFW emulator model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Base name for output files (.dill and .ckpt). "
            "If not specified, saves to the default location in the "
            "emulate directory."
        ),
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=100_000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n_valid",
        type=int,
        default=10_000,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--n_radial",
        type=int,
        default=100,
        help="Number of radial points per sample",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
        help="Total number of epochs (default: 500 from config)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show summary plots and statistics after training",
    )

    args = parser.parse_args()

    output_path = args.name if args.name else DEFAULT_MODEL_PATH

    print("Training gNFW emulator")
    print(f"  output: {output_path}.dill")
    print(f"  n_train: {args.n_train}")
    print(f"  n_valid: {args.n_valid}")
    print(f"  n_radial: {args.n_radial}")
    print(f"  n_epochs: {args.n_epochs or 'default (500)'}")
    print(f"  checkpoint_every: {args.checkpoint_every}")
    print(f"  resume: {args.resume}")
    print(f"  seed: {args.seed}")
    print(f"  plot: {args.plot}")
    print()

    model, history = rerun(
        name=args.name,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_radial=args.n_radial,
        n_epochs=args.n_epochs,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        seed=args.seed,
        plot=args.plot,
    )

    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final valid loss: {history['valid_loss'][-1]:.6f}")
    print(f"  Model saved to: {output_path}.dill")


if __name__ == "__main__":
    main()
