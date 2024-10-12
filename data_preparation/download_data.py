import requests
import tarfile
import os
import hashlib
from tqdm import tqdm

# Note: gpt-4o-mini was used to assist in the writing of this script


def download_tar_file(url: str, dest_path: str) -> str:
    """
    Download a tar file from a given URL and save it to a destination path.

    Args:
        url (str): URL of the tar file to download.
        dest_path (str): Destination path where the tar file will be saved.

    Returns:
        str: Path to the downloaded tar file.
    """
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\nDownloaded tar file to {dest_path}")
        return dest_path
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")


def extract_tar_file(tar_path: str, extract_path: str) -> None:
    """
    Extract a tar file to a specified directory.

    Args:
        tar_path (str): Path to the tar file to extract.
        extract_path (str): Directory to extract contents to.
    """
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted tar file to {extract_path}")


def compute_file_hash(file_path: str) -> str:
    """
    Compute the SHA-256 hash of the file.

    Args:
        file_path (str): Path to the file for which to compute the hash.

    Returns:
        str: The SHA-256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    dest_path = "../data/VOCtrainval_11-May-2012.tar"
    extract_path = "../data"
    expected_hash = "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"

    # Create the extract path directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    try:
        # Check if the file already exists
        if os.path.exists(dest_path):
            # Compute the hash of the existing file
            existing_hash = compute_file_hash(dest_path)
            if existing_hash == expected_hash:
                print("File already exists and hash matches. No need to download.")
            else:
                print("File exists but hash does not match. Downloading...")
                download_tar_file(url, dest_path)
        else:
            print(f"{dest_path} does not exist. Downloading...")
            download_tar_file(url, dest_path)

        if os.path.exists(os.path.join(extract_path, "VOCdevkit")):
            print("File already extracted. No need to extract.")
        else:
            extract_tar_file(dest_path, extract_path)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
