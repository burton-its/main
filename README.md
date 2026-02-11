# Music Auth App

## Overview
This is a small Flask app that provides user authentication: login, logout, and registration. Credentials are stored in a MySQL database and passwords are hashed with bcrypt.

## Requirements
- Python 3.10+
- MySQL



## Environment Variables
The current `config.py` reads these environment variable names: (use config_example.py, congfig.py is in 
.gitignore)

```
MYSQL_HOST=localhost
MYSQL_DB=musicdb
MYSQL_SECRET_KEY=your_secret_key
APP_USER=app_user
```



## Database Schema
The app expects a `users` table with three columns
- `id` (primary key)
- `email` (unique)
- `password_hash`

## Running the App


```
python app.py
```


## Routes
- `GET /` or `GET /login`: Show login form
- `POST /login`: Authenticate user
- `GET /logout`: Clear session
- `GET /register`: Show registration form
- `POST /register`: Create a new user
