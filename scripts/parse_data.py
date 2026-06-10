import argparse
from pathlib import Path

from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.parser import RemNoteParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse RemNote markdown data into local LlamaIndex document storage.")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional isolated output root. When set, parsed external files are written under this root and "
            "local storage is written to <output-root>/storage."
        ),
    )
    parser.add_argument("--raw-data-dir", default=None, help="Raw RemNote markdown directory to parse.")
    parser.add_argument("--parsed-pdfs-dir", default=None, help="Directory for downloaded/parsed PDFs.")
    parser.add_argument("--parsed-images-dir", default=None, help="Directory for downloaded/parsed images.")
    parser.add_argument("--parsed-texts-dir", default=None, help="Directory for downloaded/parsed HTML/text files.")
    parser.add_argument("--local-storage-dir", default=None, help="Directory for local LlamaIndex storage files.")
    return parser.parse_args()


def make_path_settings(args: argparse.Namespace) -> PathSettings:
    output_root = Path(args.output_root) if args.output_root else None
    defaults = PathSettings()

    return PathSettings(
        raw_data_dir=Path(args.raw_data_dir)
        if args.raw_data_dir
        else (output_root / "raw" / "AI Research" if output_root else defaults.raw_data_dir),
        parsed_pdfs_dir=Path(args.parsed_pdfs_dir)
        if args.parsed_pdfs_dir
        else (output_root / "parsed_pdfs" if output_root else defaults.parsed_pdfs_dir),
        parsed_images_dir=Path(args.parsed_images_dir)
        if args.parsed_images_dir
        else (output_root / "parsed_images" if output_root else defaults.parsed_images_dir),
        parsed_texts_dir=Path(args.parsed_texts_dir)
        if args.parsed_texts_dir
        else (output_root / "parsed_texts" if output_root else defaults.parsed_texts_dir),
        local_storage_dir=Path(args.local_storage_dir)
        if args.local_storage_dir
        else (output_root / "storage" if output_root else defaults.local_storage_dir),
    )


def main():
    args = parse_args()
    path_settings = make_path_settings(args)
    storage_settings = StorageSettings()

    # This is the local storage setup, comment the settings lines below to activate non-local storages
    storage_settings.document_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.index_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.vector_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.property_graph_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)

    parser = RemNoteParser(path_settings, storage_settings)
    parser.run()


if __name__ == '__main__':
    main()
