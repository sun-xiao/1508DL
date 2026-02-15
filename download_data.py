import os
import subprocess

DATA_DIR = "./data/raw"
FILE_NAME = "training.1600000.processed.noemoticon.csv"


def download_sentiment140():
    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, FILE_NAME)

    if os.path.exists(file_path):
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading Sentiment140 dataset...")

    subprocess.run(["kaggle", "datasets", "download", "-d", "kazanova/sentiment140", "-p", DATA_DIR, "--unzip"],
                   check=True)

    print("Download complete.")


if __name__ == "__main__":
    download_sentiment140()
