from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from dotenv import load_dotenv, find_dotenv
import os

env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)

sndl_password = os.getenv("SNDL_PASSWORD")

mySNdl = SNdl(LocalDirectory="data")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password=sndl_password)