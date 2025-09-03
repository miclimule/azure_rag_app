-- Table des documents
CREATE TABLE documents (
    id INT PRIMARY KEY IDENTITY(1,1),
    starting_line INT,
    book NVARCHAR(100),
    book_name NVARCHAR(255),
    chapter NVARCHAR(100),
    chapter_name NVARCHAR(MAX)
);

-- Table des utilisateurs
CREATE TABLE users (
    id INT PRIMARY KEY IDENTITY(1,1),
    nom NVARCHAR(100),
    email NVARCHAR(100),
    role NVARCHAR(50),
    date_creation DATETIME DEFAULT GETDATE()
);

-- Table des tâches
CREATE TABLE taches (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT,
    description NVARCHAR(MAX),
    date_tache DATETIME,
    statut NVARCHAR(50),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Table des capteurs environnementaux
CREATE TABLE capteurs_environnement (
    id INT PRIMARY KEY IDENTITY(1,1),
    type_capteur NVARCHAR(50),
    valeur FLOAT,
    unite NVARCHAR(20),
    date_mesure DATETIME,
    localisation NVARCHAR(100)
);

-- Table des données météo
CREATE TABLE donnees_meteo (
    id INT PRIMARY KEY IDENTITY(1,1),
    date_observation DATETIME,
    temperature FLOAT,
    humidite FLOAT,
    vent FLOAT,
    condition_meteo NVARCHAR(100),
    source NVARCHAR(100)
);

-- Table des terrains agricoles
CREATE TABLE terrains (
    id INT PRIMARY KEY IDENTITY(1,1),
    nom_terrain NVARCHAR(100),
    surface_m2 FLOAT,
    type_sol NVARCHAR(100),
    pente FLOAT,
    latitude FLOAT,
    longitude FLOAT
);

-- Table pour les embeddings de documents
CREATE TABLE document_embeddings (
    id INT PRIMARY KEY IDENTITY(1,1),
    document_id INT,
    embedding VARBINARY(MAX),
    model_version NVARCHAR(50),
    date_creation DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
