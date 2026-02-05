# Music Auth App

## Overview
This is a small Flask app that provides user authentication: login, logout, and registration. Credentials are stored in a MySQL database and passwords are hashed with bcrypt.

## Requirements
- Python 3.10+
- MySQL
- Virtual environment (recommended)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Configure environment variables.
4. Initialize your database schema.

## Environment Variables
The current `config.py` reads these environment variable names:

```
MYSQL_HOST=localhost
MYSQL_DB=musicdb
Finn3=your_secret_key
Corey=app_user
```

Note: `MYSQL_PASSWORD` is not read from a named environment variable in the current code. If you intend to supply a password, update `config.py` to use a standard key like `MYSQL_PASSWORD`.

## Database Schema
The app expects a `users` table with at least these columns:
- `id` (primary key)
- `email` (unique)
- `password_hash`

## Running the App
From the project root:

```
python app.py
```

The development server runs with `debug=True`.

## Routes
- `GET /` or `GET /login`: Show login form
- `POST /login`: Authenticate user
- `GET /logout`: Clear session
- `GET /register`: Show registration form
- `POST /register`: Create a new user
