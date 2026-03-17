from tqdm import tqdm
import requests
import tarfile
import os
import logging


def download_data(url, saved_location):
    os.makedirs(saved_location, exist_ok=True)
    file_path = os.path.join(saved_location, os.path.basename(url))

    if not os.path.exists(file_path):
        logging.info("Downloading data...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(url))

        with open(file_path, 'wb') as file:
            for data in response.iter_content(1024):
                file.write(data)
                progress_bar.update(len(data))
        
        progress_bar.close()
        
        logging.info("Successfully downloaded. File is saved in '%s'.", file_path)
    else:
        logging.info("%s is already available!", os.path.basename(url))


def extract_raw_data(raw_data_location, destination, name):
    if not os.path.exists(os.path.join(destination, name)):
        with tarfile.open(raw_data_location, "r:gz") as tar:
            total_files = sum(1 for _ in tar.getmembers())

            with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
                for member in tar:
                    tar.extract(member, destination)
                    pbar.update(1)

        logging.info("Extracted %s to %s", raw_data_location, destination)
    else:
        logging.info("%s is already extracted", name)