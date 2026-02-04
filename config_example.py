import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("", "dev")
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("")
    MYSQL_PASSWORD = os.getenv("")
    MYSQL_DB = os.getenv("MYSQL_DB")