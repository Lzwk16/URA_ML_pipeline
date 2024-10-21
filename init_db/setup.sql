-- Create projects table if it doesn't exist
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'projects') THEN
      CREATE TABLE projects (
          id SERIAL PRIMARY KEY,
          project_name VARCHAR(100),
          market_segment VARCHAR(50),
          street VARCHAR(255),
          x_coordinate NUMERIC,
          y_coordinate NUMERIC
      );
   END IF;
END
$$;

-- Create transactions table if it doesn't exist
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'private_property_transactions') THEN
      CREATE TABLE private_property_transactions (
          id SERIAL PRIMARY KEY,
          project_id INT,
          transaction_date VARCHAR(50),
          area NUMERIC,
          price NUMERIC,
          nett_price NUMERIC,
          property_type VARCHAR(100),
          tenure VARCHAR(100),
          type_of_area VARCHAR(50),
          floor_range VARCHAR(50),
          type_of_sale VARCHAR(50),
          district VARCHAR(10),
          no_of_units INT,
          FOREIGN KEY (project_id) REFERENCES projects(id)
      );
   END IF;
END
$$;