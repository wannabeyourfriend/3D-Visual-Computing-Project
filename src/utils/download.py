import argparse
import os
import requests

def download_links(file_path, download_folder):
    """
    Downloads all links from a given txt file to a specified folder.

    Args:
        file_path (str): The path to the txt file containing the links.
        download_folder (str): The path to the folder where files will be downloaded.
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created directory: {download_folder}")

    try:
        with open(file_path, 'r') as f:
            links = f.readlines()

        for link in links:
            url = link.strip().strip('"')
            if not url:
                continue

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Get filename from URL
                filename = os.path.join(download_folder, url.split('/')[-1])

                with open(filename, 'wb') as f_out:
                    for chunk in response.iter_content(chunk_size=8192):
                        f_out.write(chunk)
                print(f"Successfully downloaded {url} to {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download all links from a specified txt file.')
    parser.add_argument('file', type=str, help='The path to the .txt file containing the URLs (one per line, enclosed in "").')
    parser.add_argument('folder', type=str, help='The path to the destination folder for downloads.')

    args = parser.parse_args()

    download_links(args.file, args.folder)