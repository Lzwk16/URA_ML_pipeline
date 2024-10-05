DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'property_transactions_db') THEN
      CREATE DATABASE property_transactions_db;
   END IF;
END
$$;

\c property_transactions_db

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(100),
    market_segment VARCHAR(50),
    street VARCHAR(255),             -- Market segment (e.g., CCR, RCR, OCR)
    x_coordinate NUMERIC,
    y_coordinate NUMERIC
);

CREATE TABLE private_property_transactions (
    id SERIAL PRIMARY KEY,
    project_id INT,                   -- Add the project_id column here 
    transaction_date DATE,            -- Date of the transaction
    area NUMERIC,                     -- Land/floor area in square meters
    price NUMERIC,                    -- Transacted price
    property_type VARCHAR(100),       -- Type of the transacted property
    tenure VARCHAR(100),              -- Tenure of the property
    type_of_area VARCHAR(50),         -- Type of area (Strata, Land, Unknown)
    floor_range VARCHAR(50),          -- Floor range of the transacted unit
    type_of_sale VARCHAR(50),         -- Type of sale (1 - New Sale, 2 - Sub Sale, 3 - Resale)
    district VARCHAR(10),             -- Postal district
    no_of_units INT,                  -- Number of units in the transaction
    FOREIGN KEY (project_id) REFERENCES projects(id)
);
