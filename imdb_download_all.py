# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # IMDb: download diário de todos os ficheiros para o Volume
# MAGIC
# MAGIC Descarrega os 7 ficheiros dos datasets do IMDb (`name.basics`, `title.akas`, `title.basics`, `title.crew`, `title.episode`, `title.principals`, `title.ratings`) para `/Volumes/<catálogo>/landing/raw/imdb` e valida-os com Spark. Foi pensado para correr **todos os dias** como job agendado.
# MAGIC
# MAGIC - O download é uma tarefa de I/O: usa-se Python (`requests`, em streaming, vários ficheiros em paralelo). O **PySpark** entra na validação (cabeçalhos e, opcionalmente, contagem de linhas).
# MAGIC - **Só descarrega o que mudou:** guarda em `_manifest.json` o `ETag`/`Last-Modified` de cada ficheiro e ignora os que não mudaram desde a última execução. Os downloads vão primeiro para `.part` e só no fim ficam com o nome final, por isso uma execução interrompida nunca deixa um ficheiro truncado.
# MAGIC - **Free Edition:** o compute pode não aceder a `datasets.imdbws.com` (como acontece com o TMDB). Se for o caso, o notebook pára com uma mensagem clara; agendar não resolve isso, e terás de usar o `download_imdb_all.py` no teu computador.
# MAGIC - Datasets do IMDb: apenas para uso pessoal e não comercial. Alguns ficheiros são grandes (principals, akas, name.basics); confirma o espaço disponível.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

IMDB_FILES = [
    "name.basics.tsv.gz",
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz",
]

dbutils.widgets.text("catalog", "auto", "Catálogo (vazio/auto = detetar o catálogo do utilizador)")
dbutils.widgets.text("files", "all", "Ficheiros (all ou lista separada por vírgulas)")
dbutils.widgets.dropdown("overwrite", "nao", ["nao", "sim"], "Forçar novo download")
dbutils.widgets.text("parallel", "3", "Downloads em paralelo")
dbutils.widgets.dropdown("count_rows", "nao", ["nao", "sim"], "Contar linhas na validação (lento)")
dbutils.widgets.text("dest_dir", "", "Pasta de destino (vazio = Volume)")
dbutils.widgets.text("base_url", "https://datasets.imdbws.com", "URL base")

CATALOG    = dbutils.widgets.get("catalog").strip()
FORCE      = dbutils.widgets.get("overwrite") == "sim"
PARALLEL   = max(1, int(dbutils.widgets.get("parallel")))
COUNT_ROWS = dbutils.widgets.get("count_rows") == "sim"
BASE_URL   = dbutils.widgets.get("base_url").strip().rstrip("/")
DEST_DIR   = dbutils.widgets.get("dest_dir").strip() or f"/Volumes/{CATALOG}/landing/raw/imdb"
MANIFEST_PATH = f"{DEST_DIR}/_manifest.json"
HEADERS    = {"User-Agent": "Mozilla/5.0 (projeto-universitario)"}

_files_arg = dbutils.widgets.get("files").strip()
SELECTED = IMDB_FILES if _files_arg.lower() == "all" else [f.strip() for f in _files_arg.split(",") if f.strip()]
_unknown = [f for f in SELECTED if f not in IMDB_FILES]
if _unknown:
    raise ValueError(f"Ficheiros desconhecidos: {_unknown}. Válidos: {IMDB_FILES}")

print(f"Origem: {BASE_URL}")
print(f"Destino: {DEST_DIR}")
print(f"Ficheiros ({len(SELECTED)}): {SELECTED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Funções de download

# COMMAND ----------

def host_reachable(url):
    try:
        socket.gethostbyname(urlparse(url).hostname)
        return True
    except OSError:
        return False


def load_manifest():
    """Estado da última execução: ficheiro -> {etag, last_modified, size}."""
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def remote_meta(url):
    """ETag, Last-Modified e tamanho anunciados pelo servidor (None quando não existem)."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=30, headers=HEADERS)
        r.raise_for_status()
        return {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                "size": int(r.headers.get("Content-Length", 0)) or None}
    except Exception:
        return {"etag": None, "last_modified": None, "size": None}


def is_unchanged(name, meta, have_size, manifest):
    """True se o ficheiro local corresponde à versão atual do servidor."""
    if have_size is None:
        return False
    prev = manifest.get(name)
    if prev is None:  # ficheiro sem histórico (ex.: carregado à mão): confia no tamanho
        return meta["size"] is not None and have_size == meta["size"]
    if meta["etag"] or meta["last_modified"]:
        return prev.get("etag") == meta["etag"] and prev.get("last_modified") == meta["last_modified"]
    return meta["size"] is not None and have_size == meta["size"]


MANIFEST = load_manifest()
NEW_MANIFEST = {}  # preenchido pelos downloads (um valor por ficheiro)


def download_file(name):
    """Descarrega um ficheiro em streaming para o Volume. Devolve (ficheiro, estado, MB, segundos)."""
    url, dest = f"{BASE_URL}/{name}", f"{DEST_DIR}/{name}"
    tmp = dest + ".part"
    t0 = time.time()
    try:
        meta = remote_meta(url)
        have = os.path.getsize(dest) if os.path.exists(dest) else None
        if not FORCE and is_unchanged(name, meta, have, MANIFEST):
            NEW_MANIFEST[name] = {**meta, "size": have}
            return (name, "ignorado (sem alterações)", have / 1e6, 0.0)

        total, done, next_mark = meta["size"], 0, 10
        with requests.get(url, stream=True, timeout=60, headers=HEADERS) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                # r.raw: bytes tal como o servidor os envia (os .gz não podem ser alterados)
                for chunk in iter(lambda: r.raw.read(1024 * 1024), b""):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done * 100 // total >= next_mark:
                        print(f"  {name}: {done * 100 // total}% ({done / 1e6:,.0f}/{total / 1e6:,.0f} MB)")
                        next_mark = done * 100 // total + 10
        if total and done != total:
            raise IOError(f"download incompleto ({done} de {total} bytes)")

        status = "descarregado (atualizado)" if have is not None else "descarregado (novo)"
        if have is not None:
            os.remove(dest)
        dbutils.fs.mv(tmp, dest)
        NEW_MANIFEST[name] = {**meta, "size": done}
        return (name, status, done / 1e6, time.time() - t0)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return (name, f"ERRO: {type(e).__name__}: {e}", 0.0, time.time() - t0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download

# COMMAND ----------

if not host_reachable(BASE_URL):
    raise RuntimeError(
        f"{urlparse(BASE_URL).hostname} não é alcançável a partir deste compute (típico da Free Edition). "
        "Corre download_imdb_all.py no teu computador e carrega os ficheiros para o Volume (landing/raw/imdb).")

dbutils.fs.mkdirs(DEST_DIR)  # cria a pasta de destino se não existir (idempotente)

with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
    results = list(pool.map(download_file, SELECTED))

# Atualiza o manifesto (mantém as entradas de ficheiros que não foram tratados nesta execução)
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump({**MANIFEST, **NEW_MANIFEST}, f, indent=2)

display(spark.createDataFrame(results, "ficheiro string, estado string, mb double, segundos double"))

_failed = [r for r in results if r[1].startswith("ERRO")]
if _failed:
    raise RuntimeError(f"{len(_failed)} download(s) falharam: {[r[0] for r in _failed]}. "
                       "Volta a correr (os que não mudaram são ignorados).")
print(f"{len(results)} ficheiros prontos em {DEST_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validação com PySpark
# MAGIC
# MAGIC Lê cada ficheiro (TSV comprimido, sem inferência de tipos) e mostra as colunas. A contagem de linhas só corre se o widget `count_rows` for `sim` (o `title.principals` tem dezenas de milhões de linhas).

# COMMAND ----------

validation = []
for name in SELECTED:
    df = (spark.read.option("header", True).option("sep", "\t").option("quote", "")
          .csv(f"{DEST_DIR}/{name}"))
    rows = df.count() if COUNT_ROWS else None
    validation.append((name, len(df.columns), ", ".join(df.columns), rows))

display(spark.createDataFrame(validation, "ficheiro string, n_colunas int, colunas string, linhas long"))