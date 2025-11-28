----------------------------------------------------------
-- CLIMATE–HEALTH INTEGRATED DATABASE
----------------------------------------------------------

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name='ClimateHealthSystem')
    CREATE DATABASE ClimateHealthSystem;
GO

USE ClimateHealthSystem;
GO

----------------------------------------------------------
-- Unified Climate + Health Table
----------------------------------------------------------
IF OBJECT_ID('climate_health_data', 'U') IS NOT NULL 
    DROP TABLE climate_health_data;
GO

CREATE TABLE climate_health_data (
    id INT IDENTITY(1,1) PRIMARY KEY,
    date DATE NOT NULL,
    temperature FLOAT NOT NULL,
    rainfall FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    wind_speed FLOAT NOT NULL,
    elevation FLOAT NOT NULL,
    wind_pattern FLOAT NOT NULL,
    malaria_cases INT NOT NULL,
    facility_id INT NULL,
    created_at DATETIME DEFAULT GETDATE()
);
GO

----------------------------------------------------------
-- Predictions Table
----------------------------------------------------------
IF OBJECT_ID('predictions', 'U') IS NOT NULL DROP TABLE predictions;
GO

CREATE TABLE predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    date DATE NOT NULL,
    predicted_cases FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT GETDATE()
);
GO

----------------------------------------------------------
-- Insert Procedure
----------------------------------------------------------
GO
CREATE OR ALTER PROCEDURE InsertPrediction
    @date DATE,
    @predicted_cases FLOAT,
    @risk_level VARCHAR(20)
AS
BEGIN
    INSERT INTO predictions(date, predicted_cases, risk_level)
    VALUES (@date, @predicted_cases, @risk_level);
END;
GO
