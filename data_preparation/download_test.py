import requests
import tarfile
import os
import hashlib
from tqdm import tqdm

# Note: gpt-4o-mini was used to assist in the writing of this script


# https://stackoverflow.com/a/39225272
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


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

        # Send a request to get the file size for progress tracking
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        # Download the file with a progress bar
        with open(dest_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        print("Download completed successfully.")
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
    url = 'https://www.dropbox.com/scl/fi/jsumicca8xqj9qwjaw8ac/test.tar?rlkey=m44x5olxh8v6jwu27n54nk796&st=ofto2v33&dl=1'
    dest_path = "../data/test.tar"
    extract_path = "../test"
    expected_hash = "f08582b1935816c5eab3bbb1eb6d06201a789eaa173cdf1cf400c26f0cac2fb3"

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
