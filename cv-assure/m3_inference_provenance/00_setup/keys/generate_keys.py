"""Generate an Ed25519 key pair for local M3 provenance experiments."""

from argparse import ArgumentParser
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keys(output_dir: str | Path) -> tuple[Path, Path]:
    """Write a PEM-encoded private/public key pair to output_dir."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    private_path = directory / "private_key.pem"
    public_path = directory / "public_key.pem"

    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where private_key.pem and public_key.pem are written.",
    )
    args = parser.parse_args()
    private_path, public_path = generate_keys(args.output_dir)
    print(f"Generated {private_path}")
    print(f"Generated {public_path}")


if __name__ == "__main__":
    main()
