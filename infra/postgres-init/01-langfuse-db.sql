-- Give Langfuse its own database.
--
-- Langfuse previously shared the application's `evalrag` database, so its
-- migrations ran against the same schema as our tables. Runs only on first
-- initialisation of an empty data volume.
CREATE DATABASE langfuse OWNER evalrag;
