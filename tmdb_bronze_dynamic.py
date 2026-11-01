# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # TMDB → Bronze (geração dinâmica)
# MAGIC
# MAGIC Obtém dados da **API REST do TMDB** e cria as tabelas **bronze** a partir de uma **configuração de endpoints** (`ENDPOINTS`). O nome da tabela, as colunas e o schema são gerados a partir da configuração: para acrescentar um endpoint basta acrescentar uma entrada.
# MAGIC
# MAGIC | Modo | O que faz | Quando usar |
# MAGIC |---|---|---|
# MAGIC | `api` | Chama a API REST a partir do Databricks | Compute com acesso à internet |
# MAGIC | `volume` | Lê os `.jsonl` descarregados localmente (`tmdb_download.py`) para o Volume | **Free Edition** (sem acesso a `api.themoviedb.org`) |
# MAGIC | `auto` | Usa `api` se o TMDB for alcançável, senão `volume` | Por defeito |
# MAGIC
# MAGIC As tabelas criadas têm os **mesmos nomes e colunas** que as pipelines silver do notebook principal esperam (`tmdb_movie_lists`, `tmdb_genres`, `tmdb_movie_details`).
# MAGIC
# MAGIC **Nota:** se correres este notebook, a parte BRONZE da célula TMDB do notebook principal fica redundante (e o seu `require_files("tmdb")` falha em modo `api`); corre apenas a parte SILVER.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "abd_project", "Catálogo")
dbutils.widgets.text("bronze_schema", "bronze", "Schema bronze")
dbutils.widgets.dropdown("mode", "auto", ["auto", "api", "volume"], "Modo de obtenção")
dbutils.widgets.text("secret_scope", "tmdb", "Secret scope (modo api)")
dbutils.widgets.text("secret_key", "api-key", "Secret key (modo api)")
dbutils.widgets.text("language", "pt-PT", "Idioma")
dbutils.widgets.text("max_pages", "5", "Páginas por lista (20 filmes/página)")
dbutils.widgets.text("details_limit", "200", "Detalhe: nº de filmes mais populares")
dbutils.widgets.text("extra_ids_limit", "1000", "Detalhe: nº de ids extra (bronze.ml_links)")

CATALOG         = dbutils.widgets.get("catalog").strip()
BRONZE          = dbutils.widgets.get("bronze_schema").strip()
MODE_REQUESTED  = dbutils.widgets.get("mode")
SECRET_SCOPE    = dbutils.widgets.get("secret_scope")
SECRET_KEY      = dbutils.widgets.get("secret_key")
LANGUAGE        = dbutils.widgets.get("language")
MAX_PAGES       = int(dbutils.widgets.get("max_pages"))
DETAILS_LIMIT   = int(dbutils.widgets.get("details_limit"))
EXTRA_IDS_LIMIT = int(dbutils.widgets.get("extra_ids_limit"))

BASE_URL = "https://api.themoviedb.org/3"
TMDB_HOST = "api.themoviedb.org"
VOLUME_DIR = f"/Volumes/{CATALOG}/bronze/raw/tmdb"   # ficheiros do modo 'volume'

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DOS ENDPOINTS  (é aqui que se acrescentam novos dados)
#   table       tabela bronze de destino (vários endpoints podem escrever na mesma tabela)
#   path        caminho da API; '{id}' marca um endpoint de detalhe (um pedido por id)
#   result_key  chave do JSON com a lista de itens (None = a resposta inteira é um item)
#   paginated   True => pede as páginas 1..max_pages
#   extra       colunas fixas acrescentadas a cada linha (ex.: nome da lista)
#   id_col      nome da coluna com o id do filme (endpoints de detalhe)
#   ids         origem dos ids do detalhe (ver resolve_ids)
# ------------------------------------------------------------------------------
ENDPOINTS = [
    {"table": "tmdb_movie_lists", "path": "/movie/popular", "result_key": "results",
     "paginated": True, "extra": {"list_name": "popular"}},
    {"table": "tmdb_movie_lists", "path": "/movie/top_rated", "result_key": "results",
     "paginated": True, "extra": {"list_name": "top_rated"}},
    {"table": "tmdb_movie_lists", "path": "/movie/now_playing", "result_key": "results",
     "paginated": True, "extra": {"list_name": "now_playing"}},
    # Exemplo de novo endpoint (descomentar):
    # {"table": "tmdb_movie_lists", "path": "/trending/movie/week", "result_key": "results",
    #  "paginated": True, "extra": {"list_name": "trending_week"}},
    {"table": "tmdb_genres", "path": "/genre/movie/list", "result_key": "genres"},
    {"table": "tmdb_movie_details", "path": "/movie/{id}", "id_col": "movie_id",
     "ids": {"top_from_table": "tmdb_movie_lists", "limit": DETAILS_LIMIT,
             "extra_table": "ml_links", "extra_col": "tmdbId", "extra_limit": EXTRA_IDS_LIMIT}},
]

# Ficheiro do Volume correspondente a cada tabela (modo 'volume')
VOLUME_FILES = {
    "tmdb_movie_lists": "movie_lists.jsonl",
    "tmdb_genres": "genres.jsonl",
    "tmdb_movie_details": "movie_details.jsonl",
}

print(f"Destino: {CATALOG}.{BRONZE} | {len(ENDPOINTS)} endpoints configurados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Geração dinâmica dos schemas

# COMMAND ----------

def columns_of(ep):
    """Colunas (nome, tipo) da tabela bronze, geradas a partir da configuração do endpoint."""
    cols = [(name, "string") for name in ep.get("extra", {})]
    if ep.get("paginated"):
        cols.append(("page", "int"))
    if ep.get("id_col"):
        cols.append((ep["id_col"], "long"))
    cols.append(("payload", "string"))
    return cols


def ddl_of(cols):
    return ", ".join(f"`{name}` {dtype}" for name, dtype in cols)


# Agrupa os endpoints por tabela e garante que partilham o mesmo schema
TABLES = {}  # tabela -> {"cols": [...], "endpoints": [...]}
for _ep in ENDPOINTS:
    _cols = columns_of(_ep)
    _entry = TABLES.setdefault(_ep["table"], {"cols": _cols, "endpoints": []})
    if _entry["cols"] != _cols:
        raise ValueError(f"Endpoints da tabela '{_ep['table']}' têm colunas diferentes: "
                         f"{_entry['cols']} vs {_cols}")
    _entry["endpoints"].append(_ep)

for _table, _entry in TABLES.items():
    print(f"{_table}: {ddl_of(_entry['cols'])}  <- {[e['path'] for e in _entry['endpoints']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Escolha do modo (api ou volume)

# COMMAND ----------

def tmdb_reachable():
    try:
        socket.gethostbyname(TMDB_HOST)
        return True
    except OSError:
        return False


if MODE_REQUESTED == "auto":
    MODE = "api" if tmdb_reachable() else "volume"
    print(f"Modo auto -> '{MODE}'" + ("" if MODE == "api" else f" ({TMDB_HOST} não é alcançável a partir deste compute)"))
elif MODE_REQUESTED == "api" and not tmdb_reachable():
    raise RuntimeError(
        f"{TMDB_HOST} não é alcançável a partir deste compute (típico da Free Edition). "
        "Descarrega os dados localmente com tmdb_download.py, carrega-os para o Volume e usa o modo 'volume'.")
else:
    MODE = MODE_REQUESTED
    print(f"Modo: {MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Obtenção dos dados (modo `api`)

# COMMAND ----------

_session = None


def get_session():
    """Sessão HTTP criada só quando é preciso; o token vem do secret scope (ou de TMDB_TOKEN)."""
    global _session
    if _session is None:
        try:
            token = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
        except Exception:
            token = os.environ.get("TMDB_TOKEN")
        if not token:
            raise RuntimeError(f"Token em falta: cria o secret '{SECRET_SCOPE}/{SECRET_KEY}' ou define TMDB_TOKEN.")
        _session = requests.Session()
        _session.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json"})
    return _session


def tmdb_get(path, params=None, retries=4):
    """GET com repetição em caso de limite de pedidos (429) ou erro de servidor (5xx)."""
    session = get_session()
    params = {"language": LANGUAGE, **(params or {})}
    for attempt in range(retries):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def make_row(ep, page, item_id, item):
    """Linha na ordem das colunas geradas por columns_of."""
    row = list(ep.get("extra", {}).values())
    if ep.get("paginated"):
        row.append(page)
    if ep.get("id_col"):
        row.append(item_id)
    row.append(json.dumps(item, ensure_ascii=False))
    return tuple(row)


def fetch_paginated(ep):
    rows = []
    for page in range(1, MAX_PAGES + 1):
        data = tmdb_get(ep["path"], {"page": page})
        rows += [make_row(ep, page, None, item) for item in data[ep["result_key"]]]
        if page >= data.get("total_pages", page):
            break
    return rows


def fetch_single(ep):
    data = tmdb_get(ep["path"])
    items = data[ep["result_key"]] if ep.get("result_key") else [data]
    return [make_row(ep, None, None, item) for item in items]


def resolve_ids(ep, collected):
    """Ids para o detalhe: os mais populares das listas já obtidas + ids extra de uma tabela bronze."""
    spec = ep["ids"]
    popularity = {}
    for row in collected.get(spec["top_from_table"], []):
        movie = json.loads(row[-1])  # o payload é sempre a última coluna
        popularity[movie["id"]] = max(popularity.get(movie["id"], 0), movie.get("popularity", 0))
    ids = sorted(popularity, key=popularity.get, reverse=True)[:spec.get("limit", 200)]

    if spec.get("extra_table"):
        try:
            col = spec["extra_col"]
            extra = (spark.table(f"{CATALOG}.{BRONZE}.{spec['extra_table']}")
                     .where(F.col(col).rlike("^[0-9]+$"))
                     .select(F.col(col).cast("long").alias("id"))
                     .orderBy("id")
                     .limit(spec.get("extra_limit", 1000))
                     .collect())
            ids += [r.id for r in extra]
            print(f"  + {len(extra)} ids extra de {spec['extra_table']}.{col}")
        except Exception as e:
            print(f"  (ids extra ignorados: {type(e).__name__})")
    return list(dict.fromkeys(ids))  # sem repetidos, mantém a ordem


def fetch_details(ep, collected):
    ids = resolve_ids(ep, collected)
    failures = []

    def one(movie_id):
        try:
            return make_row(ep, None, movie_id, tmdb_get(ep["path"].format(id=movie_id)))
        except Exception as e:
            failures.append((movie_id, str(e)))
            return None

    get_session()  # valida o token antes de lançar as threads
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(one, ids) if r]
    for movie_id, err in failures[:10]:
        print(f"  Falhou {movie_id}: {err}")
    return rows


def fetch_endpoint(ep, collected):
    if "{id}" in ep["path"]:
        return fetch_details(ep, collected)
    if ep.get("paginated"):
        return fetch_paginated(ep)
    return fetch_single(ep)


collected = {}  # tabela -> linhas (o detalhe depende das listas, por isso a ordem dos ENDPOINTS conta)
if MODE == "api":
    for ep in ENDPOINTS:
        rows = fetch_endpoint(ep, collected)
        collected.setdefault(ep["table"], []).extend(rows)
        print(f"{ep['path']:<24} -> {ep['table']}: {len(rows):,} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Criação das tabelas bronze

# COMMAND ----------

summary = []
for table, entry in TABLES.items():
    ddl = ddl_of(entry["cols"])
    full = f"{CATALOG}.{BRONZE}.{table}"

    if MODE == "api":
        rows = collected.get(table, [])
        if not rows:
            raise ValueError(f"Nenhuma linha obtida para {table}")
        df = spark.createDataFrame(rows, ddl)
        origin = "api:" + ",".join(e["path"] for e in entry["endpoints"])
    else:
        path = f"{VOLUME_DIR}/{VOLUME_FILES[table]}"
        try:
            dbutils.fs.ls(path)
        except Exception:
            raise RuntimeError(
                f"Ficheiro em falta no modo 'volume': {path}. "
                "Corre o tmdb_download.py localmente e carrega os .jsonl para o Volume, "
                "ou usa o modo 'api' num compute com acesso à internet.")
        df = spark.read.schema(ddl).json(path)  # schema explícito, gerado da configuração
        origin = path

    df = (df.withColumn("_source", F.lit("tmdb"))
            .withColumn("_source_file", F.lit(origin))
            .withColumn("_ingested_at", F.current_timestamp()))

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE}")
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full))
    spark.sql(f"COMMENT ON TABLE {full} IS 'TMDB bronze ({MODE}): {origin}'")

    n = spark.table(full).count()
    summary.append((full, MODE, n))
    print(f"{full}: {n:,} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verificação

# COMMAND ----------

display(spark.createDataFrame(summary, "tabela string, modo string, linhas long"))

# COMMAND ----------

for table in TABLES:
    print(f"\n{table}")
    display(spark.table(f"{CATALOG}.{BRONZE}.{table}").limit(3))