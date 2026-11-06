# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
displayHTML("""<div style="background:#000; padding:48px 56px; border-left:14px solid #19E06E;">
<div style="display:inline-block; background:#19E06E; padding:8px 14px; color:#fff;
    font-family:Arial,sans-serif; font-size:11px; font-weight:800; letter-spacing:2.5px;
    text-transform:uppercase;">Analyzing Big Data</div>
<div style="margin-top:28px; font-family:'Segoe UI',Helvetica,Arial,sans-serif;
    font-size:54px; font-weight:800; line-height:1; letter-spacing:-2px; color:white;">
Os Melhores Filmes e Séries</div>
<div style="margin-top:18px; font-family:'Segoe UI',Helvetica,Arial,sans-serif;
    font-size:19px; font-weight:300; letter-spacing:0.5px; color:#D1D5DB;">
IMDb + TMDB + MovieLens &nbsp;|&nbsp; TP2 - Ano letivo 2026/2027</div>
<div style="margin-top:50px; font-family:'Segoe UI',Helvetica,Arial,sans-serif;
    font-size:15px; font-weight:600; letter-spacing:0.8px; color:white;">
20251694: Francisco Monteiro<br>
20251843: José Grave<br>
2025XXXX: Luís Pousada<br>
2024XXXX: Tiago Felgueira</div></div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Definição do problema
# MAGIC
# MAGIC **_"Quais são os melhores filmes e séries de todos os tempos?"_**
# MAGIC **_"Que fatores os tornam no que são?"_**
# MAGIC
# MAGIC Estas perguntas dividem críticos, amigos e colegas de grupo há décadas. Apesar de discussões longas e intensas, nunca se chegou a acordo, em parte porque cada pessoa tende a favorecer as suas próprias preferências.
# MAGIC
# MAGIC Para lhes responder com dados, o grupo faz uma análise exploratória da **qualidade dos conteúdos, medida principalmente pela nota do IMDb**, e estuda os fatores que podem explicar essa qualidade e que devem ser tidos em conta ao recomendar um filme ou uma série: tipo de conteúdo, género, década, duração, popularidade (número de votos) e, nas séries, o número de temporadas e de episódios.
# MAGIC
# MAGIC O IMDb é um dos conjuntos de dados mais conhecidos em Data Science e é a **fonte principal** deste projeto. Duas fontes complementares enriquecem os filmes:
# MAGIC - **MovieLens:** avaliações de utilizadores, uma segunda opinião que permite verificar se a nota do IMDb é consistente;
# MAGIC - **TMDB:** orçamento, receita e idioma original.
# MAGIC
# MAGIC Os dados são organizados numa estrutura **Bronze / Silver / Gold** que suporta a análise deste projeto e pode ser reutilizada por outros analistas.
# MAGIC
# MAGIC ## Arquitetura: Data Lakehouse com Medallion Architecture (aula #07)
# MAGIC
# MAGIC ```
# MAGIC  Fontes (descarregadas localmente, porque a Free Edition não acede à internet)
# MAGIC     IMDb (TSV)            TMDB (JSON)          MovieLens (CSV)
# MAGIC     fonte principal       complementar         complementar
# MAGIC          \                    |                    /
# MAGIC           v                   v                   v
# MAGIC   BRONZE   Volume /Volumes/abd_project/bronze/raw/...   ficheiros originais
# MAGIC            tabelas bronze.*   dados brutos + histórico de leituras (_batch_id)
# MAGIC          |
# MAGIC          v   validação das features
# MAGIC   SILVER   tabelas silver.*   dados limpos, tipados e integrados (um título por linha)
# MAGIC          |
# MAGIC          v
# MAGIC   GOLD     tabelas gold.*     datamart: melhores títulos e fatores de qualidade
# MAGIC ```
# MAGIC
# MAGIC | Camada | O que guarda | Exemplos neste projeto |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | Dados tal como vieram da fonte + lote e data de cada leitura | `bronze.imdb_title_basics`, `bronze.ml_ratings` |
# MAGIC | **Silver** | Dados validados, limpos, tipados e integrados por chave | `silver.imdb_titles`, `silver.titles_unified` |
# MAGIC | **Gold** | Datamart para responder às perguntas de negócio | `gold.dm_titles`, `gold.top_titles`, `gold.rating_by_factor` |
# MAGIC
# MAGIC **Dados estruturados:** notas, votos, duração, temporadas, orçamento, receita. **Dados semi-estruturados e não estruturados:** o JSON do TMDB, as sinopses e as tags de texto livre do MovieLens.
# MAGIC
# MAGIC **Como se ligam as fontes:** todos os títulos do IMDb têm um identificador `tconst` (ex.: `tt0111161`). O ficheiro `links.csv` do MovieLens liga cada filme ao seu `tconst` e ao id do TMDB.
# MAGIC
# MAGIC **Ordem de execução:** correr as secções por ordem (1 a 6). Os ficheiros têm de estar no Volume antes da secção 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup
# MAGIC
# MAGIC - **1.1 Bibliotecas:** imports e versões das ferramentas usadas.
# MAGIC - **1.2 Variáveis globais:** nome do catálogo e das camadas, caminhos dos ficheiros, schemas e limiares. Ficam todas aqui para que uma alteração (por exemplo, o nome do catálogo) se faça num só sítio.
# MAGIC - **1.3 Funções auxiliares:** pequenas funções usadas várias vezes ao longo do notebook.

# COMMAND ----------

# ==============================================================================
# 1.1 BIBLIOTECAS
# ==============================================================================
# No Databricks, o PySpark, o pandas e o matplotlib já vêm instalados.
# Se faltasse alguma biblioteca, instalava-se numa célula própria com:  %pip install <nome>
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import matplotlib.pyplot as plt
import pandas as pd
import pyspark
from pyspark.sql import functions as F

print("Python     :", sys.version.split()[0])
print("PySpark    :", pyspark.__version__)
print("Spark      :", spark.version)   # a SparkSession 'spark' é criada automaticamente pelo Databricks
print("pandas     :", pd.__version__)
print("matplotlib :", version("matplotlib"))

# COMMAND ----------

# ==============================================================================
# 1.2 VARIÁVEIS GLOBAIS
# ==============================================================================
# Widgets: parâmetros que aparecem no topo do notebook e se podem alterar sem mexer no código
dbutils.widgets.text("catalog", "abd_project", "Catálogo")
dbutils.widgets.dropdown("bronze_mode", "append", ["append", "overwrite"], "Escrita na bronze")

CATALOG     = dbutils.widgets.get("catalog")
BRONZE_MODE = dbutils.widgets.get("bronze_mode")   # append = cada execução acrescenta um novo lote
BATCH_ID    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")   # identifica a leitura desta execução

# ---------------------------- Camadas e caminhos ------------------------------
BRONZE = "bronze"
SILVER = "silver"
GOLD   = "gold"
LAYERS = [BRONZE, SILVER, GOLD]

VOLUME   = "raw"
RAW_ROOT = f"/Volumes/{CATALOG}/{BRONZE}/{VOLUME}"   # o Volume com os ficheiros fica na camada bronze
SOURCES  = {
    "imdb":      f"{RAW_ROOT}/imdb",
    "tmdb":      f"{RAW_ROOT}/tmdb",
    "movielens": f"{RAW_ROOT}/movielens",
}

# ------------------------ Ficheiros de cada fonte -----------------------------
# tabela bronze -> (ficheiro no Volume, lista de colunas ou schema)
# IMDb (fonte principal): ficheiros completos, com todos os tipos de título
IMDB_FILES = {
    "imdb_title_basics":  ("title.basics.tsv.gz",
                           ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                            "startYear", "endYear", "runtimeMinutes", "genres"]),
    "imdb_title_ratings": ("title.ratings.tsv.gz",  ["tconst", "averageRating", "numVotes"]),
    "imdb_title_episode": ("title.episode.tsv.gz",  ["tconst", "parentTconst", "seasonNumber", "episodeNumber"]),
}
TMDB_FILES = {
    "tmdb_movie_lists":   ("movie_lists.jsonl",   "list_name string, page int, payload string"),
    "tmdb_genres":        ("genres.jsonl",        "payload string"),
    "tmdb_movie_details": ("movie_details.jsonl", "movie_id long, payload string"),
}
MOVIELENS_FILES = {
    "ml_ratings": ("ratings.csv", ["userId", "movieId", "rating", "timestamp"]),
    "ml_movies":  ("movies.csv",  ["movieId", "title", "genres"]),
    "ml_links":   ("links.csv",   ["movieId", "imdbId", "tmdbId"]),
    "ml_tags":    ("tags.csv",    ["userId", "movieId", "tag", "timestamp"]),
}
FILES_BY_SOURCE = {"imdb": IMDB_FILES, "tmdb": TMDB_FILES, "movielens": MOVIELENS_FILES}

# Estrutura do JSON do TMDB (coluna 'payload'), usada para o interpretar
LIST_MOVIE_SCHEMA = ("id long, title string, original_title string, original_language string, "
                     "overview string, release_date string, popularity double, "
                     "vote_average double, vote_count long")
DETAIL_SCHEMA = ("id long, imdb_id string, title string, original_title string, original_language string, "
                 "overview string, release_date string, runtime int, budget long, revenue long, "
                 "status string, popularity double, vote_average double, vote_count long, "
                 "genres array<struct<id:int,name:string>>")

# ------------------------- Tipos de conteúdo do IMDb ---------------------------
# Só entram na análise filmes e séries (ficam de fora episódios, curtas, jogos, vídeos...)
MOVIE_TYPES  = ["movie", "tvMovie"]
SERIES_TYPES = ["tvSeries", "tvMiniSeries"]

# --------------------------------- Limiares -----------------------------------
MAX_BAD_KEYS_PCT    = 0.1    # % máxima de chaves nulas/inválidas; acima disto a silver não é carregada
MAX_BAD_FEATURE_PCT = 5.0    # % de valores inválidos numa feature a partir da qual se mostra um aviso
MIN_IMDB_VOTES      = 1000   # um título só entra na análise se tiver votos suficientes ("fiável")
TOP_MIN_VOTES       = 25000  # peso dos votos na nota ponderada do ranking (ver secção 4.6)
MIN_TITLES_PER_GROUP = 30    # mínimo de títulos por grupo nas tabelas agregadas

# ---------------------------- Configurações Spark -----------------------------
# Na Free Edition (serverless) algumas configurações são geridas pelo Databricks e não se
# podem alterar; por isso tenta-se aplicar cada uma e mostra-se o resultado.
SPARK_CONFS = {
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.adaptive.enabled": "true",                # Adaptive Query Execution (AQE)
    "spark.sql.autoBroadcastJoinThreshold": "10485760",  # tabelas até 10 MB são enviadas a todos os nós
}
for conf_name, conf_value in SPARK_CONFS.items():
    try:
        spark.conf.set(conf_name, conf_value)
        print(f"[aplicada]                {conf_name} = {conf_value}")
    except Exception:
        print(f"[gerida pelo Databricks]  {conf_name}")

print(f"\nCatálogo: {CATALOG} | Volume: {RAW_ROOT} | Lote: {BATCH_ID} | Bronze em modo: {BRONZE_MODE}")

# COMMAND ----------

# ==============================================================================
# 1.3 FUNÇÕES AUXILIARES
# ==============================================================================
def tbl(layer, name):
    """Nome completo de uma tabela no Unity Catalog: catálogo.camada.tabela"""
    return f"{CATALOG}.{layer}.{name}"


def missing_files(source):
    """Lista os ficheiros esperados de uma fonte que ainda não estão no Volume."""
    present = [f.name for f in dbutils.fs.ls(SOURCES[source])]
    expected = [file_name for file_name, _ in FILES_BY_SOURCE[source].values()]
    return [file_name for file_name in expected if file_name not in present]


def read_csv_as_strings(path, columns, sep=",", quote='"', escape="\\", multiline=False):
    """Lê um CSV/TSV com schema explícito e todas as colunas como texto (string).
    Na bronze não se convertem tipos: guarda-se o dado exatamente como veio da fonte."""
    schema = ", ".join(f"`{c}` string" for c in columns)
    return (spark.read
            .schema(schema)
            .option("header", True)
            .option("sep", sep)
            .option("quote", quote)
            .option("escape", escape)
            .option("multiLine", multiline)
            .csv(path))


def add_audit_columns(df, source):
    """Acrescenta colunas de auditoria: fonte, ficheiro de origem, lote e data da leitura."""
    return (df
            .withColumn("_source", F.lit(source))
            .withColumn("_source_file", F.col("_metadata.file_path"))
            .withColumn("_batch_id", F.lit(BATCH_ID))
            .withColumn("_ingested_at", F.current_timestamp()))


def read_bronze(table_name):
    """Lê o lote mais recente de uma tabela bronze (a bronze guarda todas as leituras)."""
    df = spark.table(tbl(BRONZE, table_name))
    last_batch = df.agg(F.max("_batch_id")).first()[0]
    return df.where(F.col("_batch_id") == last_batch)


def try_cast(column, data_type):
    """Converte o tipo de uma coluna; se o valor for inválido devolve null em vez de dar erro."""
    return F.expr(f"try_cast(`{column}` AS {data_type})")


def to_date_safe(column):
    """Converte texto 'aaaa-mm-dd' em data; qualquer outro valor (vazio, inválido) fica null."""
    return F.when(F.col(column).rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(F.col(column)))


def save_table(df, layer, name, comment, mode="overwrite", cluster_by=None):
    """Grava um DataFrame como tabela Delta no Unity Catalog e devolve a tabela gravada.
    cluster_by: colunas de Liquid Clustering (organiza os ficheiros para filtrar mais depressa)."""
    full_name = tbl(layer, name)
    if cluster_by is None:
        writer = df.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        else:
            writer = writer.option("mergeSchema", "true")
        writer.saveAsTable(full_name)
    else:
        # O Liquid Clustering define-se em SQL: cria-se uma vista temporária e a tabela a partir dela
        df.createOrReplaceTempView("tmp_to_save")
        spark.sql(f"""CREATE OR REPLACE TABLE {full_name}
                      CLUSTER BY ({", ".join(cluster_by)})
                      AS SELECT * FROM tmp_to_save""")
    spark.sql(f"COMMENT ON TABLE {full_name} IS '{comment.replace(chr(39), '')}'")
    saved = spark.table(full_name)
    print(f"[{layer}] {full_name}: {saved.count():,} linhas")
    return saved


def check_unique(df, key, label):
    """Controlo de qualidade: a chave não pode ter valores repetidos."""
    total = df.count()
    distinct = df.select(key).distinct().count()
    if total != distinct:
        raise ValueError(f"{label}: a chave '{key}' tem duplicados ({total:,} linhas, {distinct:,} distintas)")
    print(f"[qualidade] {label}: chave '{key}' única ({total:,} linhas)")


print("Funções auxiliares carregadas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Preparação do ambiente
# MAGIC
# MAGIC Cria (se ainda não existirem) o catálogo, os três schemas da arquitetura Medallion (`bronze`, `silver`, `gold`), o Volume `raw` e uma pasta por fonte. Todos os comandos usam `IF NOT EXISTS`, por isso a célula pode ser corrida várias vezes sem erros.
# MAGIC
# MAGIC No fim mostra que ficheiros já estão no Volume. Na primeira execução é normal faltarem: carregam-se em **Catalog > abd_project > bronze > raw > Upload to this volume**.

# COMMAND ----------

# ==============================================================================
# 2. PREPARAÇÃO DO AMBIENTE  -  catálogo, schemas, Volume, pastas e ficheiros
# ==============================================================================
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
for layer in LAYERS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{layer}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE}.{VOLUME}")
for folder in SOURCES.values():
    dbutils.fs.mkdirs(folder)   # não faz nada se a pasta já existir

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# Que ficheiros já estão no Volume?
file_status = []
for source in SOURCES:
    present = [f.name for f in dbutils.fs.ls(SOURCES[source])]
    for file_name, _ in FILES_BY_SOURCE[source].values():
        status = "OK" if file_name in present else "EM FALTA"
        file_status.append((source, file_name, status))

display(spark.createDataFrame(file_status, "fonte string, ficheiro string, estado string"))

n_missing = len([row for row in file_status if row[2] == "EM FALTA"])
if n_missing > 0:
    print(f"Faltam {n_missing} ficheiro(s): carrega-os para o Volume antes da secção 3.")
else:
    print("Ambiente pronto: todos os ficheiros estão no Volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingestão na camada Bronze (IMDb, TMDB, MovieLens)
# MAGIC
# MAGIC Cada ficheiro é lido do Volume com **schema explícito** e gravado **tal como veio**, mais quatro colunas de auditoria: `_source`, `_source_file`, `_batch_id` e `_ingested_at`. Não se usa `inferSchema`, porque obriga o Spark a ler os dados duas vezes e pode adivinhar mal os tipos.
# MAGIC
# MAGIC A escrita é em modo **append**: cada execução acrescenta um novo lote, por isso a bronze guarda o **histórico de todas as leituras** e é possível reprocessar a silver a partir de qualquer lote. As camadas seguintes usam sempre o lote mais recente (função `read_bronze`). Para começar do zero, muda o widget `bronze_mode` para `overwrite`.
# MAGIC
# MAGIC O IMDb é o maior volume de dados do projeto: só o `title.basics` tem mais de 10 milhões de títulos e o `title.episode` cerca de 9 milhões de episódios.

# COMMAND ----------

# =========================== 3.1 BRONZE · IMDb (TSV) ==========================
# Os ficheiros do IMDb são separados por tabulação (\t) e não usam aspas; valores em falta vêm como \N.
# Têm todos os tipos de título (filmes, séries, episódios...): na bronze guarda-se tudo.
if missing_files("imdb"):
    raise FileNotFoundError(f"Faltam ficheiros do IMDb: {missing_files('imdb')}")

for table_name, (file_name, columns) in IMDB_FILES.items():
    df_raw = read_csv_as_strings(f"{SOURCES['imdb']}/{file_name}", columns, sep="\t", quote="")
    save_table(add_audit_columns(df_raw, "imdb"), BRONZE, table_name, mode=BRONZE_MODE,
               comment=f"IMDb {file_name}, valores em texto")

# COMMAND ----------

# =========================== 3.2 BRONZE · TMDB (JSON) =========================
# Alternativa: o notebook tmdb_bronze_dynamic.py cria estas mesmas tabelas a partir da API do TMDB.
if missing_files("tmdb"):
    raise FileNotFoundError(f"Faltam ficheiros do TMDB: {missing_files('tmdb')}")

for table_name, (file_name, schema) in TMDB_FILES.items():
    df_raw = spark.read.schema(schema).json(f"{SOURCES['tmdb']}/{file_name}")
    save_table(add_audit_columns(df_raw, "tmdb"), BRONZE, table_name, mode=BRONZE_MODE,
               comment=f"TMDB {file_name}: JSON bruto, um lote por leitura")

# COMMAND ----------

# ========================= 3.3 BRONZE · MovieLens (CSV) =======================
# Títulos e tags podem ter vírgulas e aspas, por isso usa-se escape='"' e multiLine.
if missing_files("movielens"):
    raise FileNotFoundError(f"Faltam ficheiros do MovieLens: {missing_files('movielens')}")

for table_name, (file_name, columns) in MOVIELENS_FILES.items():
    df_raw = read_csv_as_strings(f"{SOURCES['movielens']}/{file_name}", columns,
                                 escape='"', multiline=(table_name != "ml_ratings"))
    save_table(add_audit_columns(df_raw, "movielens"), BRONZE, table_name, mode=BRONZE_MODE,
               comment=f"MovieLens {file_name}, valores em texto")

# COMMAND ----------

# ====================== 3.4 BRONZE · histórico de leituras =====================
# Uma linha por tabela e por lote: mostra que a bronze guarda todas as leituras feitas.
all_bronze_tables = list(IMDB_FILES) + list(TMDB_FILES) + list(MOVIELENS_FILES)

batches_per_table = []
for table_name in all_bronze_tables:
    batches = (spark.table(tbl(BRONZE, table_name))
               .groupBy("_batch_id")
               .agg(F.count("*").alias("linhas"), F.max("_ingested_at").alias("lido_em"))
               .withColumn("tabela", F.lit(table_name)))
    batches_per_table.append(batches)

# Junta as tabelas de resumo numa só (union = empilhar linhas)
history = batches_per_table[0]
for batches in batches_per_table[1:]:
    history = history.unionByName(batches)

display(history.select("tabela", "_batch_id", "linhas", "lido_em").orderBy("tabela", F.desc("_batch_id")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Camada Silver e datamart de negócio
# MAGIC
# MAGIC ### 4.1 Identificação e validação das features
# MAGIC
# MAGIC Antes de carregar a silver identifica-se o **papel** de cada variável e verifica-se a qualidade do lote mais recente da bronze:
# MAGIC
# MAGIC | Papel | Significado | O que acontece se houver problemas |
# MAGIC |---|---|---|
# MAGIC | `chave` | identifica cada linha (ex.: `tconst`, `movieId`) | acima de `MAX_BAD_KEYS_PCT` a pipeline **para** |
# MAGIC | `ligacao` | liga uma tabela a outra (ex.: `parentTconst`, `imdbId`) | aviso acima de `MAX_BAD_FEATURE_PCT` |
# MAGIC | `feature` | variável usada na análise (nota, votos, ano, duração...) | aviso acima de `MAX_BAD_FEATURE_PCT` |
# MAGIC | `texto` | títulos, sinopses, tags (dados não estruturados) | só se contam os nulos |
# MAGIC
# MAGIC Para cada coluna contam-se três coisas: valores **nulos**, valores **inválidos** (não convertem para o tipo certo ou não seguem o formato esperado) e valores **fora do intervalo** (ex.: uma nota acima de 10). O relatório fica guardado em `silver.dq_feature_report`.
# MAGIC
# MAGIC _Nota (aula #04):_ os DataFrames de `DATA_TO_VALIDATE` são só **transformações**, por isso o Spark ainda não leu nada (**lazy evaluation**). Os dados só são lidos quando a função `validate_table` executa a **ação** `.first()`.

# COMMAND ----------

# ================= 4.1 IDENTIFICAÇÃO E VALIDAÇÃO DAS FEATURES ==================
# Regras por tabela: coluna, tipo esperado, papel e, opcionalmente, formato (regex) e intervalo [min, max]
FEATURE_RULES = {
    "imdb_title_basics": [
        {"col": "tconst",         "type": "string", "role": "chave",   "regex": r"^tt\d{7,}$"},
        {"col": "titleType",      "type": "string", "role": "feature"},
        {"col": "primaryTitle",   "type": "string", "role": "texto"},
        {"col": "isAdult",        "type": "int",    "role": "feature", "regex": r"^[01]$"},
        {"col": "startYear",      "type": "int",    "role": "feature", "min": 1870, "max": 2035},
        {"col": "endYear",        "type": "int",    "role": "feature", "min": 1870, "max": 2035},
        {"col": "runtimeMinutes", "type": "int",    "role": "feature", "min": 0},
        {"col": "genres",         "type": "string", "role": "feature"},
    ],
    "imdb_title_ratings": [
        {"col": "tconst",        "type": "string", "role": "chave",   "regex": r"^tt\d{7,}$"},
        {"col": "averageRating", "type": "double", "role": "feature", "min": 1, "max": 10},
        {"col": "numVotes",      "type": "bigint", "role": "feature", "min": 0},
    ],
    "imdb_title_episode": [
        {"col": "tconst",        "type": "string", "role": "chave",   "regex": r"^tt\d{7,}$"},
        {"col": "parentTconst",  "type": "string", "role": "ligacao", "regex": r"^tt\d{7,}$"},
        {"col": "seasonNumber",  "type": "int",    "role": "feature", "min": 0, "max": 1000},
        {"col": "episodeNumber", "type": "int",    "role": "feature", "min": 0},
    ],
    "tmdb_movie_details": [
        {"col": "id",                "type": "bigint", "role": "chave",   "min": 1},
        {"col": "imdb_id",           "type": "string", "role": "ligacao", "regex": r"^tt\d{7,}$"},
        {"col": "overview",          "type": "string", "role": "texto"},
        {"col": "original_language", "type": "string", "role": "feature", "regex": r"^[a-z]{2}$"},
        {"col": "budget",            "type": "bigint", "role": "feature", "min": 0},
        {"col": "revenue",           "type": "bigint", "role": "feature", "min": 0},
        {"col": "vote_average",      "type": "double", "role": "feature", "min": 0, "max": 10},
    ],
    "ml_ratings": [
        {"col": "userId",    "type": "int",    "role": "chave",   "min": 1},
        {"col": "movieId",   "type": "int",    "role": "chave",   "min": 1},
        {"col": "rating",    "type": "double", "role": "feature", "min": 0.5, "max": 5},
        {"col": "timestamp", "type": "bigint", "role": "feature", "min": 0},
    ],
    "ml_links": [
        {"col": "movieId", "type": "int",    "role": "chave",   "min": 1},
        {"col": "imdbId",  "type": "string", "role": "ligacao", "regex": r"^\d+$"},
        {"col": "tmdbId",  "type": "int",    "role": "ligacao", "min": 1},
    ],
    "ml_tags": [
        {"col": "userId",  "type": "int",    "role": "chave", "min": 1},
        {"col": "movieId", "type": "int",    "role": "chave", "min": 1},
        {"col": "tag",     "type": "string", "role": "texto"},
    ],
}

# Dados a validar: lote mais recente de cada tabela bronze (o JSON do TMDB é interpretado primeiro)
DATA_TO_VALIDATE = {
    "imdb_title_basics":  read_bronze("imdb_title_basics"),
    "imdb_title_ratings": read_bronze("imdb_title_ratings"),
    "imdb_title_episode": read_bronze("imdb_title_episode"),
    "tmdb_movie_details": (read_bronze("tmdb_movie_details")
                           .select(F.from_json("payload", DETAIL_SCHEMA).alias("d"))
                           .select("d.*")),
    "ml_ratings":         read_bronze("ml_ratings"),
    "ml_links":           read_bronze("ml_links"),
    "ml_tags":            read_bronze("ml_tags"),
}


def validate_table(table_name, df, rules):
    """Conta nulos, inválidos e valores fora do intervalo em cada coluna, numa só leitura da tabela."""
    columns = [rule["col"] for rule in rules]
    # Passa tudo a texto e trata '\N' (IMDb) e '' como valores em falta
    df = df.select([F.col(c).cast("string").alias(c) for c in columns]).replace(["\\N", ""], None)

    counts = [F.count("*").alias("total")]
    for rule in rules:
        c = rule["col"]
        value = F.col(c)
        typed = try_cast(c, rule["type"])

        # inválido: tem valor mas não converte para o tipo esperado, ou não segue o formato
        is_invalid = value.isNotNull() & typed.isNull()
        if "regex" in rule:
            is_invalid = is_invalid | (value.isNotNull() & ~value.rlike(rule["regex"]))

        # fora do intervalo [min, max]
        is_out = F.lit(False)
        if "min" in rule:
            is_out = is_out | (typed < rule["min"])
        if "max" in rule:
            is_out = is_out | (typed > rule["max"])

        counts.append(F.sum(F.when(value.isNull(), 1).otherwise(0)).alias(f"{c}__nulos"))
        counts.append(F.sum(F.when(is_invalid, 1).otherwise(0)).alias(f"{c}__invalidos"))
        counts.append(F.sum(F.when(is_out, 1).otherwise(0)).alias(f"{c}__fora"))

    result = df.agg(*counts).first()   # <- ação: só aqui o Spark lê os dados

    report = []
    total = result["total"]
    for rule in rules:
        c = rule["col"]
        role = rule["role"]
        nulls = result[f"{c}__nulos"] or 0
        invalid = result[f"{c}__invalidos"] or 0
        out = result[f"{c}__fora"] or 0
        pct_nulls = round(100 * nulls / total, 2) if total else 0.0
        pct_bad = round(100 * (invalid + out) / total, 2) if total else 0.0

        if role == "chave" and pct_nulls + pct_bad > MAX_BAD_KEYS_PCT:
            status = "FAIL"
        elif role in ("ligacao", "feature") and pct_bad > MAX_BAD_FEATURE_PCT:
            status = "WARN"
        else:
            status = "OK"
        report.append((table_name, c, role, rule["type"], total, nulls, pct_nulls, invalid, out, pct_bad, status))
    return report


dq_rows = []
for table_name, rules in FEATURE_RULES.items():
    dq_rows = dq_rows + validate_table(table_name, DATA_TO_VALIDATE[table_name], rules)

dq_report = spark.createDataFrame(
    dq_rows,
    "tabela string, coluna string, papel string, tipo string, linhas long, nulos long, pct_nulos double, "
    "invalidos long, fora_intervalo long, pct_problemas double, estado string",
).withColumn("_batch_id", F.lit(BATCH_ID))

save_table(dq_report, SILVER, "dq_feature_report", mode="append",
           comment="Validação das features da bronze antes do carregamento da silver")
display(dq_report.orderBy(F.desc("estado"), "tabela"))

n_fail = len([row for row in dq_rows if row[-1] == "FAIL"])
n_warn = len([row for row in dq_rows if row[-1] == "WARN"])
print(f"Colunas validadas: {len(dq_rows)} | avisos (WARN): {n_warn} | erros (FAIL): {n_fail}")
if n_fail > 0:
    raise ValueError("Há chaves com problemas (ver tabela acima): a silver não é carregada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 a 4.4 Silver de cada fonte
# MAGIC
# MAGIC | Fonte | Limpeza e transformação |
# MAGIC |---|---|
# MAGIC | **IMDb** | `\N` passa a `null`; ficam só filmes e séries sem conteúdo adulto; tipos convertidos com `try_cast`; géneros em lista; nota e votos juntos a cada título; nas séries, número de temporadas e de episódios a partir do `title.episode` |
# MAGIC | **TMDB** | interpretar o JSON com `from_json`, converter datas e remover duplicados |
# MAGIC | **MovieLens** | converter o `timestamp` em data, construir a chave do IMDb (`tconst`) e normalizar as tags |
# MAGIC
# MAGIC As tabelas maiores usam **Liquid Clustering** (aula #07): o Delta organiza os ficheiros pelas colunas indicadas, para que as consultas que filtram por essas colunas leiam menos dados.

# COMMAND ----------

# ============================== 4.2 SILVER · IMDb ==============================
# No IMDb, os valores em falta vêm como \N: replace passa-os a null
imdb_ratings = (read_bronze("imdb_title_ratings")
                .replace("\\N", None)
                .select("tconst",
                        try_cast("averageRating", "double").alias("imdb_rating"),
                        try_cast("numVotes", "bigint").alias("imdb_votes"))
                .dropDuplicates(["tconst"]))

imdb_titles = (read_bronze("imdb_title_basics")
               .replace("\\N", None)
               .where(F.col("titleType").isin(MOVIE_TYPES + SERIES_TYPES))       # só filmes e séries
               .where(F.col("isAdult").isNull() | (F.col("isAdult") != "1"))     # sem conteúdo adulto
               .select("tconst",
                       F.col("primaryTitle").alias("title"),
                       F.col("titleType").alias("title_type"),
                       F.when(F.col("titleType").isin(MOVIE_TYPES), "Filme")
                        .otherwise("Série").alias("content_type"),
                       try_cast("startYear", "int").alias("start_year"),
                       try_cast("endYear", "int").alias("end_year"),             # só as séries têm ano de fim
                       try_cast("runtimeMinutes", "int").alias("runtime_minutes"),
                       F.split("genres", ",").alias("genres"))   # "Drama,Romance" -> ["Drama", "Romance"]
               .dropDuplicates(["tconst"])
               .join(imdb_ratings, on="tconst", how="left"))
check_unique(imdb_titles, "tconst", "silver.imdb_titles")
imdb_titles = save_table(imdb_titles, SILVER, "imdb_titles", cluster_by=["content_type", "start_year"],
                         comment="IMDb: filmes e séries (sem conteúdo adulto) com nota e votos")

# Séries: número de temporadas e de episódios (cada linha do title.episode é um episódio)
imdb_series_stats = (read_bronze("imdb_title_episode")
                     .replace("\\N", None)
                     .select(F.col("parentTconst").alias("tconst"),
                             try_cast("seasonNumber", "int").alias("season"))
                     .groupBy("tconst")
                     .agg(F.count("*").alias("n_episodes"),
                          F.max("season").alias("n_seasons")))
save_table(imdb_series_stats, SILVER, "imdb_series_stats",
           comment="IMDb: número de temporadas e de episódios de cada série")

# Relação título-género: explode cria uma linha por cada género da lista
imdb_title_genres = imdb_titles.select("tconst", "content_type", F.explode("genres").alias("genre"))
save_table(imdb_title_genres, SILVER, "imdb_title_genres", comment="IMDb: géneros de cada título")

# COMMAND ----------

# ============================== 4.3 SILVER · TMDB ==============================
# Detalhe dos filmes: orçamento, receita, idioma original (complementa os filmes do IMDb)
tmdb_details = (read_bronze("tmdb_movie_details")
                .select(F.from_json("payload", DETAIL_SCHEMA).alias("d"))
                .select("d.*")
                .replace("", None)                                    # texto vazio passa a null
                .select(F.col("id").alias("tmdb_id"), "imdb_id", "title", "original_language", "overview",
                        to_date_safe("release_date").alias("release_date"),
                        F.col("runtime").alias("runtime_minutes"),
                        "budget", "revenue", "popularity", "vote_average", "vote_count")
                .where(F.col("tmdb_id").isNotNull())
                .dropDuplicates(["tmdb_id"]))
check_unique(tmdb_details, "tmdb_id", "silver.tmdb_movie_details")
save_table(tmdb_details, SILVER, "tmdb_movie_details",
           comment="TMDB: detalhe dos filmes (orçamento, receita, idioma, sinopse)")

# Filmes das listas do TMDB (popular, top_rated, now_playing): um filme pode estar em várias listas
tmdb_movies = (read_bronze("tmdb_movie_lists")
               .select("list_name", F.from_json("payload", LIST_MOVIE_SCHEMA).alias("m"))
               .select("list_name", "m.*")
               .groupBy(F.col("id").alias("tmdb_id"))
               .agg(F.first("title").alias("title"),
                    F.max("vote_average").alias("vote_average"),
                    F.max("vote_count").alias("vote_count"),
                    F.collect_set("list_name").alias("lists")))
check_unique(tmdb_movies, "tmdb_id", "silver.tmdb_movies")
save_table(tmdb_movies, SILVER, "tmdb_movies", comment="TMDB: filmes das listas e listas onde aparecem")

# COMMAND ----------

# =========================== 4.4 SILVER · MovieLens ============================
ml_ratings = (read_bronze("ml_ratings")
              .select(try_cast("userId", "int").alias("user_id"),
                      try_cast("movieId", "int").alias("movie_id"),
                      try_cast("rating", "double").alias("rating"),
                      F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("rated_at"))   # segundos -> data
              .where(F.col("movie_id").isNotNull() & F.col("rating").isNotNull())
              .dropDuplicates(["user_id", "movie_id", "rated_at"]))
save_table(ml_ratings, SILVER, "ml_ratings", cluster_by=["movie_id"],
           comment="MovieLens: avaliações dos utilizadores (escala 0.5 a 5)")

# Chave do IMDb: "114709" -> "tt0114709" (zeros à esquerda até 7 dígitos).
# Ids com 8 dígitos (filmes recentes) ficam como estão: o lpad cortaria o último dígito.
ml_links = (read_bronze("ml_links")
            .select(try_cast("movieId", "int").alias("movie_id"),
                    F.when(F.length("imdbId") >= 7, F.concat(F.lit("tt"), F.col("imdbId")))
                     .otherwise(F.concat(F.lit("tt"), F.lpad("imdbId", 7, "0"))).alias("tconst"),
                    try_cast("tmdbId", "int").alias("tmdb_id"))
            .where(F.col("movie_id").isNotNull())
            .dropDuplicates(["movie_id"]))
check_unique(ml_links, "movie_id", "silver.ml_links")
save_table(ml_links, SILVER, "ml_links", comment="MovieLens: ligação movie_id - tconst (IMDb) - tmdb_id (TMDB)")

# Tags de texto livre (dados não estruturados): minúsculas e sem espaços a mais
ml_tags = (read_bronze("ml_tags")
           .select(try_cast("userId", "int").alias("user_id"),
                   try_cast("movieId", "int").alias("movie_id"),
                   F.lower(F.trim("tag")).alias("tag"),
                   F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("tagged_at"))
           .where(F.col("tag").isNotNull() & (F.col("tag") != "")))
save_table(ml_tags, SILVER, "ml_tags", comment="MovieLens: tags de texto livre normalizadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Silver integrada: `silver.titles_unified`
# MAGIC
# MAGIC Junta tudo numa só tabela, com **um título do IMDb por linha**:
# MAGIC - todos os filmes e séries do IMDb, com nota e votos (fonte principal);
# MAGIC - nas séries, o número de temporadas e de episódios;
# MAGIC - nos filmes que existem no MovieLens, a nota dos utilizadores do MovieLens e os dados do TMDB (orçamento, receita, idioma).
# MAGIC
# MAGIC As tabelas pequenas (ligações do MovieLens, TMDB, notas do MovieLens) entram com **Broadcast Join** (aula #04): em vez de redistribuir a tabela grande pela rede (*shuffle*), o Spark envia uma cópia da tabela pequena para cada nó. O `explain` mostra o plano de execução com o `BroadcastHashJoin`.
# MAGIC
# MAGIC No TMDB, orçamento, receita e votos iguais a `0` significam "desconhecido", por isso passam a `null`.

# COMMAND ----------

# ========================== 4.5 SILVER INTEGRADA ===============================
titles       = spark.table(tbl(SILVER, "imdb_titles"))
series_stats = spark.table(tbl(SILVER, "imdb_series_stats"))

# Ligação MovieLens -> IMDb: um movie_id por tconst
links = (spark.table(tbl(SILVER, "ml_links"))
         .where(F.col("tconst").isNotNull())
         .dropDuplicates(["tconst"]))

# Nota média e número de avaliações de cada filme no MovieLens
ml_stats = (spark.table(tbl(SILVER, "ml_ratings"))
            .groupBy("movie_id")
            .agg(F.round(F.avg("rating"), 3).alias("ml_avg_rating"),
                 F.count("*").alias("ml_num_ratings")))

tmdb_selected = (spark.table(tbl(SILVER, "tmdb_movie_details"))
                 .select("tmdb_id", "original_language", "overview",
                         F.when(F.col("vote_count") > 0, F.col("vote_average")).alias("tmdb_vote_average"),
                         F.when(F.col("budget") > 0, F.col("budget")).alias("budget"),
                         F.when(F.col("revenue") > 0, F.col("revenue")).alias("revenue")))

titles_unified = (titles
                  .join(series_stats, on="tconst", how="left")                   # + temporadas/episódios
                  .join(F.broadcast(links), on="tconst", how="left")             # + ligação ao MovieLens
                  .join(F.broadcast(tmdb_selected), on="tmdb_id", how="left")    # + TMDB
                  .join(F.broadcast(ml_stats), on="movie_id", how="left")        # + notas MovieLens
                  .select("tconst", "title", "title_type", "content_type",
                          "start_year", "end_year",
                          (F.floor(F.col("start_year") / 10) * 10).alias("decade"),   # 1994 -> 1990
                          "runtime_minutes", "genres", "imdb_rating", "imdb_votes",
                          "n_seasons", "n_episodes",
                          "movie_id", "tmdb_id",
                          (F.col("ml_avg_rating") * 2).alias("ml_rating_10"),         # escala 0-5 -> 0-10
                          "ml_num_ratings", "tmdb_vote_average",
                          "budget", "revenue", "original_language", "overview"))

titles_unified.explain(mode="formatted")   # plano de execução: procurar "BroadcastHashJoin"

check_unique(titles_unified, "tconst", "silver.titles_unified")
titles_unified = save_table(titles_unified, SILVER, "titles_unified", cluster_by=["content_type", "decade"],
                            comment="Filmes e séries do IMDb, enriquecidos com MovieLens e TMDB")

# Quantos títulos há de cada tipo, e quantos têm dados das fontes complementares?
display(titles_unified.groupBy("content_type")
        .agg(F.count("*").alias("titulos"),
             F.count("imdb_rating").alias("com_nota_imdb"),
             F.count("n_seasons").alias("com_temporadas"),
             F.count("ml_rating_10").alias("com_nota_movielens"),
             F.count("budget").alias("com_orcamento_tmdb")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.6 Datamart de negócio (camada Gold)
# MAGIC
# MAGIC **`gold.dm_titles`** é a tabela principal do datamart: um título por linha, com a nota do IMDb (medida principal de qualidade), os fatores em estudo e as respetivas classes (duração, popularidade, temporadas). A coluna `is_reliable` indica se o título tem votos suficientes (`MIN_IMDB_VOTES`) para entrar na análise.
# MAGIC
# MAGIC A partir dela criam-se as tabelas que respondem às perguntas:
# MAGIC
# MAGIC | Tabela | Pergunta |
# MAGIC |---|---|
# MAGIC | `gold.top_titles` | Quais são os melhores filmes e séries de todos os tempos? |
# MAGIC | `gold.rating_by_factor` | Como varia a nota com a década, o género, a duração, a popularidade e as temporadas? |
# MAGIC | `gold.rating_drivers` | Que fatores numéricos estão mais associados a uma nota alta? |
# MAGIC | `gold.imdb_vs_other_sources` | A nota do IMDb é consistente com o MovieLens e o TMDB? |
# MAGIC | `gold.roi_by_genre` | Nos filmes, a qualidade acompanha o retorno financeiro? |
# MAGIC
# MAGIC **Nota ponderada (ranking):** ordenar só pela nota favorece títulos com poucos votos (uma nota 9,8 com 1 200 votos vale menos do que 9,3 com 2 milhões). Por isso o ranking usa a média ponderada que o IMDb usou no seu Top 250:
# MAGIC
# MAGIC `nota ponderada = v / (v + m) × R + m / (v + m) × C`
# MAGIC
# MAGIC onde `R` é a nota do título, `v` o número de votos, `m` = `TOP_MIN_VOTES` e `C` a nota média de todos os títulos do mesmo tipo. Com poucos votos, a nota aproxima-se da média; com muitos, aproxima-se da nota real.

# COMMAND ----------

# ======================== 4.6 DATAMART DE NEGÓCIO (GOLD) =======================
titles_unified = spark.table(tbl(SILVER, "titles_unified"))

runtime   = F.col("runtime_minutes")
votes     = F.col("imdb_votes")
seasons   = F.col("n_seasons")
rating    = F.col("imdb_rating")

dm_titles = titles_unified.select(
    "tconst", "title", "title_type", "content_type",
    F.col("start_year").alias("year"), "decade", "end_year",
    "runtime_minutes", "genres",
    F.expr("try_element_at(genres, 1)").alias("main_genre"),       # primeiro género (null se não houver)
    F.when(F.col("genres").isNotNull(), F.size("genres")).alias("n_genres"),
    rating.alias("rating_imdb"),
    votes.alias("imdb_votes"),
    F.round(F.log10(votes), 2).alias("log_votes"),                  # popularidade em escala logarítmica
    "n_seasons", "n_episodes",
    (F.col("end_year") - F.col("start_year") + 1).alias("series_years"),   # anos em exibição (séries terminadas)

    # Classes de cada fator (o número no início serve para ordenar)
    F.when(runtime < 60, "1: <60 min")
     .when(runtime < 90, "2: 60-89 min")
     .when(runtime < 120, "3: 90-119 min")
     .when(runtime < 150, "4: 120-149 min")
     .when(runtime >= 150, "5: 150+ min").alias("runtime_class"),
    F.when(votes < 1000, "1: <1k")
     .when(votes < 10000, "2: 1k-10k")
     .when(votes < 100000, "3: 10k-100k")
     .when(votes < 1000000, "4: 100k-1M")
     .when(votes >= 1000000, "5: 1M+").alias("votes_class"),
    F.when(seasons == 1, "1: 1 temporada")
     .when(seasons <= 3, "2: 2-3 temporadas")
     .when(seasons <= 6, "3: 4-6 temporadas")
     .when(seasons >= 7, "4: 7+ temporadas").alias("seasons_class"),
    F.when(rating >= 8, "1: Excelente (8+)")
     .when(rating >= 7, "2: Bom (7-8)")
     .when(rating >= 5.5, "3: Médio (5.5-7)")
     .when(rating < 5.5, "4: Fraco (<5.5)").alias("quality_class"),

    # Fontes complementares (só filmes do MovieLens)
    F.col("ml_rating_10").alias("rating_movielens"),
    F.col("tmdb_vote_average").alias("rating_tmdb"),
    F.round(rating - F.col("ml_rating_10"), 2).alias("gap_imdb_vs_movielens"),
    "budget", "revenue",
    F.round(F.col("revenue") / F.col("budget"), 2).alias("roi"),         # retorno: receita / orçamento
    "original_language",
    F.coalesce((votes >= MIN_IMDB_VOTES) & F.col("decade").isNotNull(), F.lit(False)).alias("is_reliable"),
)

check_unique(dm_titles, "tconst", "gold.dm_titles")
dm_titles = save_table(dm_titles, GOLD, "dm_titles", cluster_by=["content_type", "decade"],
                       comment="Datamart de filmes e séries: nota IMDb e fatores de qualidade")
reliable = dm_titles.where(F.col("is_reliable"))

# COMMAND ----------

# ----------- 1) Melhores filmes e séries de todos os tempos (Spark SQL) -----------
# Window function (aula #06): ROW_NUMBER() numera os títulos dentro de cada tipo (Filme / Série)
spark.sql(f"""
    CREATE OR REPLACE TABLE {tbl(GOLD, 'top_titles')} AS
    WITH type_means AS (
        SELECT content_type, AVG(rating_imdb) AS mean_rating
        FROM {tbl(GOLD, 'dm_titles')}
        WHERE is_reliable
        GROUP BY content_type
    ),
    scored AS (
        SELECT d.content_type, d.title, d.year, d.main_genre, d.rating_imdb, d.imdb_votes,
               ROUND(d.imdb_votes / (d.imdb_votes + {TOP_MIN_VOTES}) * d.rating_imdb
                     + {TOP_MIN_VOTES} / (d.imdb_votes + {TOP_MIN_VOTES}) * m.mean_rating, 3) AS weighted_rating
        FROM {tbl(GOLD, 'dm_titles')} d
        JOIN type_means m ON d.content_type = m.content_type
        WHERE d.is_reliable
    ),
    ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY content_type ORDER BY weighted_rating DESC) AS ranking
        FROM scored
    )
    SELECT * FROM ranked WHERE ranking <= 25
""")
print(f"[gold] {tbl(GOLD, 'top_titles')}: top 25 de filmes e de séries (Spark SQL)")

# ----------- 2) Nota média por fator (DataFrame API) ---------------------------
# Para cada fator: número de títulos e nota média do IMDb, por tipo de conteúdo
FACTORS = {
    "década":       "decade",
    "género":       "genre",            # cada título conta em todos os seus géneros (explode)
    "duração":      "runtime_class",
    "popularidade": "votes_class",
    "temporadas":   "seasons_class",    # só séries
    "idioma":       "original_language",  # só filmes com TMDB
}

factor_tables = []
for factor_name, column in FACTORS.items():
    data = reliable
    if column == "genre":
        data = reliable.withColumn("genre", F.explode("genres"))
    summary = (data
               .where(F.col(column).isNotNull())
               .groupBy("content_type", column)
               .agg(F.count("*").alias("n_titles"),
                    F.round(F.avg("rating_imdb"), 2).alias("avg_imdb_rating"),
                    F.round(F.stddev("rating_imdb"), 2).alias("std_imdb_rating"))
               .where(F.col("n_titles") >= MIN_TITLES_PER_GROUP)
               .select(F.lit(factor_name).alias("factor"), "content_type",
                       F.col(column).cast("string").alias("value"),
                       "n_titles", "avg_imdb_rating", "std_imdb_rating"))
    factor_tables.append(summary)

rating_by_factor = factor_tables[0]
for summary in factor_tables[1:]:
    rating_by_factor = rating_by_factor.unionByName(summary)
save_table(rating_by_factor, GOLD, "rating_by_factor", comment="Nota média do IMDb por fator e tipo de conteúdo")

# ----------- 3) Fatores numéricos associados à nota (correlação) -------------
# Correlação de Pearson entre a nota do IMDb e cada fator (-1 a 1). Mede associação, não causalidade.
DRIVERS = {
    "Filme": ["year", "runtime_minutes", "log_votes", "n_genres", "budget", "rating_movielens"],
    "Série": ["year", "runtime_minutes", "log_votes", "n_genres", "n_seasons", "n_episodes", "series_years"],
}
driver_rows = []
for content_type, drivers in DRIVERS.items():
    subset = reliable.where(F.col("content_type") == content_type)
    correlations = subset.agg(*[F.corr("rating_imdb", d).alias(d) for d in drivers]).first()
    for d in drivers:
        value = correlations[d]
        driver_rows.append((content_type, d, round(value, 3) if value is not None else None))

rating_drivers = spark.createDataFrame(driver_rows, "content_type string, factor string, correlation double")
save_table(rating_drivers, GOLD, "rating_drivers", comment="Correlação entre a nota do IMDb e cada fator numérico")

# ----------- 4) A nota do IMDb é consistente com as outras fontes? (filmes) ---
imdb_vs_other_sources = (reliable
                         .where(F.col("rating_movielens").isNotNull())
                         .groupBy("decade")
                         .agg(F.count("*").alias("n_movies"),
                              F.round(F.avg("rating_imdb"), 2).alias("avg_imdb_rating"),
                              F.round(F.avg("rating_movielens"), 2).alias("avg_movielens_rating"),
                              F.round(F.avg("rating_tmdb"), 2).alias("avg_tmdb_rating"),
                              F.round(F.corr("rating_imdb", "rating_movielens"), 3).alias("corr_imdb_movielens"))
                         .where(F.col("n_movies") >= MIN_TITLES_PER_GROUP))
save_table(imdb_vs_other_sources, GOLD, "imdb_vs_other_sources",
           comment="Filmes: nota IMDb vs MovieLens (0-10) e TMDB, por década")

# ----------- 5) Qualidade e retorno financeiro por género (filmes com TMDB) ---
roi_by_genre = (reliable
                .where(F.col("roi").isNotNull())
                .withColumn("genre", F.explode("genres"))
                .groupBy("genre")
                .agg(F.count("*").alias("n_movies"),
                     F.round(F.avg("rating_imdb"), 2).alias("avg_imdb_rating"),
                     F.round(F.percentile_approx("roi", 0.5), 2).alias("median_roi"))
                .where(F.col("n_movies") >= 5))
save_table(roi_by_genre, GOLD, "roi_by_genre", comment="Filmes: nota IMDb e retorno (receita / orçamento) por género")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Análise e visualização
# MAGIC
# MAGIC As consultas usam apenas a camada gold (o datamart), que já está agregada e pronta para análise.
# MAGIC
# MAGIC ### 5.1 Os melhores filmes e séries de todos os tempos

# COMMAND ----------

display(spark.table(tbl(GOLD, "top_titles")).where("content_type = 'Filme'").orderBy("ranking"))

# COMMAND ----------

display(spark.table(tbl(GOLD, "top_titles")).where("content_type = 'Série'").orderBy("ranking"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Que fatores explicam a qualidade?

# COMMAND ----------

# Nota média por fator: comparar filmes e séries em cada classe
display(spark.table(tbl(GOLD, "rating_by_factor")).orderBy("factor", "content_type", "value"))

# COMMAND ----------

# Gráfico: nota média por década, filmes vs séries
# O toPandas() só é aceitável aqui porque a tabela gold é pequena (uma linha por década e tipo)
pdf = (spark.table(tbl(GOLD, "rating_by_factor"))
       .where(F.col("factor") == "década")
       .orderBy("value")
       .toPandas())

fig, ax = plt.subplots(figsize=(9, 4.5))
for content_type in ["Filme", "Série"]:
    data = pdf[pdf["content_type"] == content_type]
    ax.plot(data["value"], data["avg_imdb_rating"], marker="o", label=content_type)
ax.set_title("Nota média do IMDb por década")
ax.set_xlabel("Década")
ax.set_ylabel("Nota média (0-10)")
ax.grid(alpha=0.3)
ax.legend()
plt.show()

# COMMAND ----------

# Correlação de cada fator numérico com a nota do IMDb
display(spark.table(tbl(GOLD, "rating_drivers")).orderBy("content_type", F.desc("correlation")))

# COMMAND ----------

# Gráfico: correlações lado a lado para filmes e séries
pdf_drivers = spark.table(tbl(GOLD, "rating_drivers")).toPandas()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, content_type in zip(axes, ["Filme", "Série"]):
    data = pdf_drivers[pdf_drivers["content_type"] == content_type].sort_values("correlation")
    ax.barh(data["factor"], data["correlation"])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title(f"{content_type}: correlação com a nota IMDb")
    ax.set_xlim(-1, 1)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 A nota do IMDb é consistente com outras fontes? (filmes)

# COMMAND ----------

display(spark.table(tbl(GOLD, "imdb_vs_other_sources")).orderBy("decade"))

# COMMAND ----------

display(spark.table(tbl(GOLD, "roi_by_genre")).orderBy(F.desc("avg_imdb_rating")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tecnologias Big Data: evidências
# MAGIC
# MAGIC - **Lazy evaluation e plano de execução** (aula #04): o `explain` mostra o plano que o Spark otimizou antes de ler os dados (procurar `BroadcastHashJoin` e `AdaptiveSparkPlan`).
# MAGIC - **Delta Lake** (aula #07): cada escrita cria uma nova versão da tabela (`DESCRIBE HISTORY`) e é possível consultar uma versão antiga (**time travel**).
# MAGIC - **Liquid Clustering:** `DESCRIBE DETAIL` mostra as colunas pelas quais a tabela está organizada.

# COMMAND ----------

# Plano de execução: filtro por tipo e década (beneficia do Liquid Clustering) e agregação por género
(spark.table(tbl(GOLD, "dm_titles"))
      .where((F.col("content_type") == "Série") & (F.col("decade") == 2010))
      .groupBy("main_genre")
      .agg(F.avg("rating_imdb").alias("avg_imdb_rating"))
      .explain(mode="formatted"))

# COMMAND ----------

# Delta Lake: histórico de versões de uma tabela bronze (uma versão por cada leitura em append)
display(spark.sql(f"DESCRIBE HISTORY {tbl(BRONZE, 'imdb_title_ratings')}")
        .select("version", "timestamp", "operation"))

# Time travel: número de linhas na primeira versão da tabela e agora
first_version = spark.sql(f"SELECT COUNT(*) AS n FROM {tbl(BRONZE, 'imdb_title_ratings')} VERSION AS OF 0").first()["n"]
current       = spark.table(tbl(BRONZE, "imdb_title_ratings")).count()
print(f"Linhas na versão 0: {first_version:,} | linhas agora: {current:,}")

# COMMAND ----------

# Liquid Clustering: colunas de organização do datamart
display(spark.sql(f"DESCRIBE DETAIL {tbl(GOLD, 'dm_titles')}")
        .select("name", "format", "numFiles", "sizeInBytes", "clusteringColumns"))

# COMMAND ----------

# Inventário final: todas as tabelas criadas e o número de linhas de cada uma
inventory = []
for layer in LAYERS:
    for row in spark.sql(f"SHOW TABLES IN {CATALOG}.{layer}").collect():
        if not row.isTemporary:
            n_rows = spark.table(tbl(layer, row.tableName)).count()
            inventory.append((layer, row.tableName, n_rows))

display(spark.createDataFrame(inventory, "camada string, tabela string, linhas long").orderBy("camada", "tabela"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusões
# MAGIC
# MAGIC **Os melhores de todos os tempos:** _(comentar o top 25 de filmes e de séries: géneros, décadas e tipos que dominam.)_
# MAGIC
# MAGIC **Fatores de qualidade:** _(com base em `rating_by_factor` e `rating_drivers`: o que distingue os títulos com nota alta? diferenças entre filmes e séries?)_
# MAGIC
# MAGIC **Consistência da nota do IMDb:** _(a nota do IMDb acompanha a do MovieLens e a do TMDB? em que décadas divergem mais?)_
# MAGIC
# MAGIC **Processo e tecnologia:** _(bronze com histórico de leituras e reprocessamento, validação das features, datamart gold; Apache Spark no Databricks Free Edition, Delta Lake, Unity Catalog, Liquid Clustering, Broadcast Join, Spark SQL.)_
# MAGIC
# MAGIC **Limitações:** a nota do IMDb reflete quem vota no IMDb (não é uma medida objetiva); o MovieLens-small cobre cerca de 9 mil filmes e nenhuma série; o TMDB só tem detalhe para os filmes descarregados; a correlação mede associação e não causalidade.
# MAGIC
# MAGIC **Trabalho futuro:** acrescentar elenco e realizadores (`title.principals`, `title.crew`) como fatores, analisar o texto das sinopses e das tags, e treinar um modelo para prever a nota a partir dos fatores do datamart.
