from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.parser import RemNoteParser


def main():
    path_settings = PathSettings()
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
