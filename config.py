import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("Finn2", "dev")
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("Corey")
    MYSQL_PASSWORD = os.getenv("Finn1")
    MYSQL_DB = os.getenv("MYSQL_DB")