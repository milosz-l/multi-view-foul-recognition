from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from dotenv import load_dotenv, find_dotenv
import os
import zipfile

# Get the current username
username = os.environ['USER']

# Use the username to construct paths
output_path = f"/net/tscratch/people/{username}/data"

def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)

sndl_password = os.getenv("SNDL_PASSWORD")

mySNdl = SNdl(LocalDirectory="data")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password=sndl_password)

os.makedirs(os.path.join(output_path, "Train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "Valid"), exist_ok=True)
os.makedirs(os.path.join(output_path, "Test"), exist_ok=True)
os.makedirs(os.path.join(output_path, "Chall"), exist_ok=True)

unzip_file('data/mvfouls/train.zip', os.path.join(output_path, "Train"))
unzip_file('data/mvfouls/valid.zip', os.path.join(output_path, "Valid"))
unzip_file('data/mvfouls/test.zip', os.path.join(output_path, "Test"))
unzip_file('data/mvfouls/challenge.zip', os.path.join(output_path, "Chall"))

os.remove('data/mvfouls/train.zip')
os.remove('data/mvfouls/valid.zip')
os.remove('data/mvfouls/test.zip')
os.remove('data/mvfouls/challenge.zip')