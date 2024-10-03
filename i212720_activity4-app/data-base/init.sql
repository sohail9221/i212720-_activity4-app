CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sepal_length FLOAT,
    sepal_width FLOAT,
    petal_length FLOAT,
    petal_width FLOAT,
    prediction VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
