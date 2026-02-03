CREATE DATABASE IF NOT EXISTS musicapp;
USE musicapp;

CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE user_preferences (
  user_id INT PRIMARY KEY,
  favorite_genres JSON NULL,
  favorite_artists JSON NULL,
  CONSTRAINT fk_user_pref 
  FOREIGN KEY (user_id) REFERENCES users(id) 
);