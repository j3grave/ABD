-- =============================================================
-- ABD Project - Arquitetura Medallion: setup do catálogo
-- Cria o catálogo do projeto e os schemas das três camadas
-- =============================================================

CREATE CATALOG IF NOT EXISTS abd_project
  COMMENT 'Catálogo do projeto ABD - arquitetura medallion';

CREATE SCHEMA IF NOT EXISTS abd_project.bronze
  COMMENT 'Camada Bronze - dados brutos tal como chegam da fonte';

CREATE SCHEMA IF NOT EXISTS abd_project.silver
  COMMENT 'Camada Silver - dados limpos, validados e deduplicados';

CREATE SCHEMA IF NOT EXISTS abd_project.gold
  COMMENT 'Camada Gold - tabelas agregadas prontas para análise';
