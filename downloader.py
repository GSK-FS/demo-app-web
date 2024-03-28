import requests
import os
import threading
import time

class DownloadThread(threading.Thread):
    def __init__(self, url, start_byte, end_byte, output_file, progress_callback):
        super().__init__()
        self.url = url
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.output_file = output_file
        self.progress_callback = progress_callback

    def run(self):
        headers = {'Range': f'bytes={self.start_byte}-{self.end_byte}'}
        response = requests.get(self.url, headers=headers, stream=True)
        with open(self.output_file, 'r+b') as file:
            file.seek(self.start_byte)
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                self.progress_callback(len(chunk))

def download_file(url, num_threads=8):
    # Send a HEAD request to get the file size
    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))

    # Create a file to save the downloaded content
    output_file = os.path.basename(url)
    with open(output_file, 'wb') as file:
        file.write(b'\x00' * file_size)

    # Initialize progress variables
    total_downloaded = 0
    lock = threading.Lock()

    # Progress callback function
    def progress_callback(chunk_size):
        nonlocal total_downloaded
        with lock:
            total_downloaded += chunk_size
            progress = (total_downloaded / file_size) * 100
            print(f"Downloaded: {total_downloaded} bytes ({progress:.2f}%)", end='\r')

    # Calculate the range for each thread
    chunk_size = file_size // num_threads
    threads = []
    for i in range(num_threads):
        start_byte = i * chunk_size
        end_byte = start_byte + chunk_size - 1
        if i == num_threads - 1:
            end_byte = file_size - 1
        thread = DownloadThread(url, start_byte, end_byte, output_file, progress_callback)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("\nDownload complete")

if __name__ == "__main__":
    url = input("Enter the download link: ")
    download_file(url)
