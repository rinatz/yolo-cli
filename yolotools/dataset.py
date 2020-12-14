from pathlib import Path
import os
from typing import Union
from urllib.parse import urlparse

import requests
from rich.progress import Progress


def download(url: str, dest: Union[str, Path]):
    dest = Path(dest).expanduser()
    os.makedirs(dest, exist_ok=True)

    basename = Path(urlparse(url).path).name
    filename = dest.joinpath(basename)

    if filename.is_file():
        return filename

    response = requests.get(url, stream=True)

    total = int(response.headers["Content-Length"])
    chunk_size = 1024

    data = response.iter_content(chunk_size=chunk_size)

    with filename.open("wb") as f, Progress() as progress:
        description = f"[green]Download[/green] [b]{basename} => {dest}[/b]"
        task = progress.add_task(description, total=total)

        while not progress.finished:
            f.write(next(data))
            progress.update(task, advance=chunk_size)

    return filename
