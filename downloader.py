import aiofiles
import aiohttp
import asyncio
import datetime
import hashlib
import logging

from dataclasses import dataclass, field

from pathlib import Path
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn, TaskID

# Some basic logging.
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    filename=f"./logs/download_{current_datetime}.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FILES = {
    ".":[
        "LICENSE",
        "USE_POLICY.md",
        "tokenizer.model",
        "tokenizer_checklist.chk",
    ],
    "llama-2-7b":[
        "consolidated.00.pth",
        "params.json",
        "checklist.chk"
    ],
    "llama-2-7b-chat":[
        "consolidated.00.pth",
        "params.json",
        "checklist.chk",
    ],
    "llama-2-13b":[
        "consolidated.00.pth",
        "consolidated.01.pth",
        "params.json",
        "checklist.chk",
    ],
    "llama-2-13b-chat":[
        "consolidated.00.pth",
        "consolidated.01.pth",
        "params.json",
        "checklist.chk",
    ],
    "llama-2-70b":[
        "consolidated.00.pth",
        "consolidated.01.pth",
        "consolidated.02.pth",
        "consolidated.03.pth",
        "consolidated.04.pth",
        "consolidated.05.pth",
        "consolidated.06.pth",
        "consolidated.07.pth",
        "params.json",
        "checklist.chk",
    ],
    "llama-2-70b-chat":[
        "consolidated.00.pth",
        "consolidated.01.pth",
        "consolidated.02.pth",
        "consolidated.03.pth",
        "consolidated.04.pth",
        "consolidated.05.pth",
        "consolidated.06.pth",
        "consolidated.07.pth",
        "params.json",
        "checklist.chk",
    ]
}

@dataclass(order=True)
class FileInfo:
    """Individual file information."""
    sort_index: int = field(init=False, repr=False)

    file_url: str
    file_name: str
    file_path: str
    progress_task: TaskID = TaskID(0)

    def __str__(self):
        return f"{self.file_path}"

    def __getitem__(self, item):
        return getattr(self, item)

user_entry = {
    "7B":"llama-2-7b",
    "13B":"llama-2-13b",
    "70B":"llama-2-70b",
    "7B-chat":"llama-2-7b-chat",
    "13B-chat":"llama-2-13b-chat",
    "70B-chat":"llama-2-70b-chat"
}

def verify_md5(chk_folder: str = "", file_info: list[FileInfo] = []) -> None:
    """Verify a files md5."""
    if file_info.__len__() > 0:
        retrieve = file_info
    else:
        retrieve = Path(chk_folder).glob('**/*')

    check_files = [file.__str__() for file in retrieve if ".chk" in file.__str__()]

    md_list: dict[Path, str] = {}
    for file in check_files:
        file = file.replace("\\", "/")
        with open(file, "r", encoding="utf-8") as chk_f:
            md_details = chk_f.read().split("\n")
        path = file[:file.rfind("/")]
        for md in md_details:
            if md.__len__() > 0:
                file_name = md[:md.rfind("  ")]
                file_path = f"{path}/{md[md.rfind('  ') + 2:]}"
                md_list[file_path] = file_name

    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        *Progress.get_default_columns()
    ) as progress:        
        for file_path, md_origin in md_list.items():
            try:
                with open(file_path, "rb") as f:
                    md5_hash = hashlib.md5()
                    task = progress.add_task(
                        f"[yellow]Checking [b]{file_path[file_path.rfind('/') + 1:]}[/b]",
                        total = Path(file_path).stat().st_size
                    )
                    while chunk := f.read(4096):
                        md5_hash.update(chunk)
                        progress.update(task, advance=len(chunk))
            except Exception as e:
                logging.error(f"{e}", exc_info=True)

            if md5_hash.hexdigest() == md_origin:
                info = f"{file_path} is OK"
                logging.info(info)
            else:
                info = f"{file_path} | {md_origin} != {md5_hash.hexdigest()}"
                logging.error(info)
            print(info)

def download_files(models:list[str], url:str, base_path:Path) -> list[FileInfo]:
    """Download files"""
    async def _file_downloader(progress: Progress, file_info: FileInfo) -> None:
        f_name = file_info.file_name
        header = {
            "User-Agent": "Wget/1.21.4",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "Keep-Alive",
            "Referer": "https://download.llamameta.net/"
        }
        async with aiohttp.ClientSession() as session: #, asyncio.Semaphore(4):   ############
            async with session.get(file_info.file_url, headers = header) as response, \
            aiofiles.open(file_info.file_path, "wb") as local_file:
                if response.status == 200:
                    f_size = response.content_length
                    progress.update(
                        file_info.progress_task,
                        total = f_size
                    )
                    progress.start_task(file_info.progress_task)
                    async for data in response.content.iter_chunked(1024):
                        await local_file.write(data)
                        progress.update(
                            file_info.progress_task,
                            advance=len(data),
                            description = f"[yellow]Downloading [b]{f_name}[/b]"
                        )
                    description = f"[green][b]Finished {f_name}[/b] | {f_size}"
                    logging.info(
                        f"{response.status} | {f_name} | {f_size} bytes completed."
                    )
                else:
                    logging.error(
                        f"{response.status} | {f_name}\n{response.reason}\n{file_info.file_url}"
                    )
                    description = f"[red]Unable to download [b]{f_name}[/b] - {response.status}"
            progress.update(
                file_info.progress_task,
                description = description
            )

    async def _download(progress: Progress, file_list: list[FileInfo]):
        await asyncio.gather(
            *(_file_downloader(progress, file) for file in file_list)
        )

    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        *Progress.get_default_columns()
    ) as progress:
        file_list: list[FileInfo] = []
        for model in FILES.keys():
            if model in models:
                for file in FILES[model]:
                    if model == ".":
                        file_path = f"{base_path}/{file}"
                    else:
                        file_path = f"{base_path}/{model}/{file}"
                    file_list += [
                        FileInfo(
                            file_url  = f"{url.format(f'{model}/{file}')}",
                            file_name = file,
                            file_path = file_path,
                            progress_task = progress.add_task(
                                f"[blue]Queued [b]{file}[/b]",
                                total=100,
                                start=False
                            )
                        )
                    ]
        asyncio.run(_download(progress, file_list))
    return file_list

def main_prompt():
    logging.debug("Downloading the files.")
    base = Path(__file__).cwd()
    # Alternatively, just paste your url as the value below ↓:
    url = input("Enter the URL from email: ")
    url = url.replace("*", "{0}")
    prompt = (
        "Enter the list of models to download without spaces like so ↓ \n"
      + "(7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all: "
    )
    models = input(f"{prompt}")
    if models.__len__() == 0:
        models_list = user_entry.keys()
    else:
        models_list = models.split(",")

    download_list = ["."]
    for model in models_list:
        if model not in user_entry.keys():
            raise ValueError(f"{model} is an unavailable option.")
        else:
            download_list += [user_entry[model]]
            if not Path(f"{base}/{user_entry[model]}").is_dir():
                Path(f"{base}/{user_entry[model]}").mkdir()
    
    downloaded_files = download_files(download_list, url, base)
    verify_md5(file_info = downloaded_files)


if __name__ == "__main__":
    main_prompt()
