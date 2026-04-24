from typing import Any, List, Dict
import skops.io as sio
import polars as pl


def load_assets(paths: List[str]) -> List[Any]:
    assets = []
    for path in paths:
        ext = path.split(".")[-1].lower()
        match ext:
            case "parquet":
                assets.append(pl.read_parquet(path))
            case "skops":
                trusted_types = sio.get_untrusted_types(file=path)
                assets.append(sio.load(path, trusted=trusted_types))
            case _:
                raise ValueError(f"Unsupported format '.{ext}' in path: {path}")
    return assets[0] if len(assets) == 1 else assets


def save_assets(asset_path_map: Dict[Any, str]) -> None:

    for path, asset in asset_path_map.items():
        ext = path.split(".")[-1].lower()

        if isinstance(asset, pl.DataFrame):
            match ext:
                case "parquet":
                    asset.write_parquet(path)
                case _:
                    raise ValueError(
                        f"Unsupported format '.{ext}' for a Polars DataFrame."
                    )
        else:
            match ext:
                case "skops":
                    sio.dump(asset, path)
                case _:
                    raise ValueError(f"Unsupported format '.{ext}' for a model object.")
