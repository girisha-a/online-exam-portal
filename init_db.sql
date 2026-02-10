CREATE DATABASE online_exam;

USE online_exam;

CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  password VARCHAR(100),
  role VARCHAR(20)
);

CREATE TABLE exams (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(100),
  duration INT
);

CREATE TABLE questions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  exam_id INT,
  question TEXT,
  opt_a VARCHAR(200),
  opt_b VARCHAR(200),
  opt_c VARCHAR(200),
  opt_d VARCHAR(200),
  correct CHAR(1)
);

CREATE TABLE results (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  score INT,
  risk_level VARCHAR(20),
  prediction VARCHAR(50)
);
