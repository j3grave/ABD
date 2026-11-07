# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Projeto de grupo – Filmes: IMDb + TMDB + MovieLens
# MAGIC
# MAGIC **Pergunta de negócio:** as notas dos filmes diferem entre o IMDb, o TMDB e os utilizadores do MovieLens? Que fatores (género, década, idioma, orçamento) explicam essas diferenças?
# MAGIC
# MAGIC **Equipa:** _(nomes)_  |  **Data:** _(data de entrega)_
# MAGIC
# MAGIC ## Arquitetura (Lakehouse + Medallion)
# MAGIC
# MAGIC ```
# MAGIC  Fontes (descarregadas localmente, porque a Free Edition não acede à internet)
# MAGIC     IMDb (TSV)   TMDB (JSON)   MovieLens (CSV)
# MAGIC          \            |            /
# MAGIC           v           v           v
# MAGIC   BRONZE  <catalogo>.bronze
# MAGIC             Volume raw   /Volumes/<catalogo>/bronze/raw/{imdb,tmdb,movielens}   (ficheiros originais)
# MAGIC                |  leitura com schema explícito
# MAGIC                v
# MAGIC             tabelas Delta bronze.*   dados brutos em string + metadados de auditoria
# MAGIC          |
# MAGIC          v
# MAGIC   SILVER  <catalogo>.silver.*   limpeza, tipagem, deduplicação e integração das 3 fontes
# MAGIC          |
# MAGIC          v
# MAGIC   GOLD    <catalogo>.gold.*     tabelas agregadas que respondem à pergunta de negócio
# MAGIC ```
# MAGIC
# MAGIC | Camada | Regra | Exemplos neste projeto |
# MAGIC |---|---|---|
# MAGIC | Bronze (Volume) | Ficheiros originais, sem alterações | `.tsv.gz`, `.csv`, `.jsonl` no Volume `bronze.raw` |
# MAGIC | Bronze (tabelas) | Fiel à origem; só se acrescentam metadados (`_source`, `_source_file`, `_ingested_at`) | `bronze.imdb_title_basics`, `bronze.tmdb_movie_details` |
# MAGIC | Silver | Dados limpos, tipados, sem duplicados e integrados por chave | `silver.imdb_movies`, `silver.movies_unified` |
# MAGIC | Gold | Pronto para análise e decisão | `gold.ratings_by_decade`, `gold.roi_by_genre` |
# MAGIC
# MAGIC **Dados estruturados:** notas, votos, orçamento, receita, duração. **Não estruturados:** o JSON bruto do TMDB, as sinopses e as tags de texto livre do MovieLens.
# MAGIC
# MAGIC **Chaves de integração:** o `links.csv` do MovieLens liga `movieId` ↔ `imdbId` ↔ `tmdbId`. O IMDb usa `tconst` (`tt0114709`), construído a partir do `imdbId` com `lpad` a 7 dígitos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup inicial
# MAGIC
# MAGIC A célula seguinte é **sempre a primeira a correr**. Valida e prepara:
# MAGIC 1. versão do Python e bibliotecas (instala as que faltarem);
# MAGIC 2. sessão Spark, catálogo, schemas (`bronze`, `silver`, `gold`) e o Volume `raw` dentro da bronze;
# MAGIC 3. pastas de cada fonte no Volume (cria-as se não existirem) e presença dos ficheiros;
# MAGIC 4. permissões de escrita (Volume e Delta);
# MAGIC 5. configurações Spark recomendadas e funções auxiliares que impõem as boas práticas de PySpark.

# COMMAND ----------

# ==============================================================================
# SETUP INICIAL  -  validação de ambiente, criação de caminhos e boas práticas
# ==============================================================================
import importlib
import importlib.metadata as md
import subprocess
import sys
import uuid

from pyspark.sql import functions as F

# ------------------------------- 1. Parâmetros --------------------------------
dbutils.widgets.text("catalog", "abd_project", "Catálogo")
dbutils.widgets.dropdown("auto_install", "sim", ["sim", "nao"], "Instalar libs em falta")

CATALOG      = dbutils.widgets.get("catalog").strip()
AUTO_INSTALL = dbutils.widgets.get("auto_install") == "sim"

BRONZE, SILVER, GOLD = "bronze", "silver", "gold"
LAYERS   = [BRONZE, SILVER, GOLD]
VOLUME   = "raw"
RAW_ROOT = f"/Volumes/{CATALOG}/{BRONZE}/{VOLUME}"  # o Volume vive na camada bronze
SOURCES  = {s: f"{RAW_ROOT}/{s}" for s in ("tmdb", "imdb", "movielens")}

EXPECTED_FILES = {
    "tmdb":      ["movie_lists.jsonl", "genres.jsonl", "movie_details.jsonl"],
    "imdb":      ["title.basics.movies.tsv.gz", "title.ratings.movies.tsv.gz"],
    "movielens": ["ratings.csv", "movies.csv", "links.csv", "tags.csv"],
}
REQUIRED_LIBS = [("pandas", "pandas"), ("matplotlib", "matplotlib")]  # (import, pip)
MIN_PYTHON    = (3, 9)

# ------------------------- 2. Infraestrutura de validação ---------------------
results = []  # (verificação, estado, detalhe)


def run_check(name, fn, critical=True):
    """Executa uma verificação. Falha => FAIL (crítica) ou WARN (não crítica)."""
    try:
        results.append((name, "OK", str(fn() or "")))
    except Exception as e:
        results.append((name, "FAIL" if critical else "WARN", f"{type(e).__name__}: {e}"))


# --------------------------- 3. Python e bibliotecas --------------------------
def check_python():
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        raise RuntimeError(f"Python {v.major}.{v.minor} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_pyspark():
    import pyspark
    return f"pyspark {pyspark.__version__}"


def make_lib_check(import_name, pip_name):
    def _check():
        try:
            version = md.version(pip_name)
        except md.PackageNotFoundError:
            if not AUTO_INSTALL:
                raise RuntimeError(f"'{pip_name}' não está instalada (auto_install = nao)")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            importlib.invalidate_caches()
            version = f"{md.version(pip_name)} (instalada agora)"
        importlib.import_module(import_name)
        return f"{pip_name} {version}"
    return _check


run_check("Python", check_python)
run_check("Biblioteca: pyspark", check_pyspark)
for _imp, _pip in REQUIRED_LIBS:
    run_check(f"Biblioteca: {_pip}", make_lib_check(_imp, _pip))


# ----------------- 4. Spark, catálogo, schemas, volume e pastas ---------------
def check_spark():
    if spark.range(5).count() != 5:
        raise RuntimeError("A sessão Spark devolveu um resultado inesperado")
    return f"Spark {spark.version}"


def check_catalog():
    catalogs = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
    if CATALOG not in catalogs:
        raise RuntimeError(f"Catálogo '{CATALOG}' não existe. Disponíveis: {catalogs}")
    return f"Catálogo '{CATALOG}' encontrado"


def ensure_schema(schema):
    def _ensure():
        existed = spark.sql(f"SHOW SCHEMAS IN {CATALOG} LIKE '{schema}'").count() > 0
        if not existed:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
        return f"{CATALOG}.{schema} " + ("já existia" if existed else "criado")
    return _ensure


def ensure_volume():
    existed = spark.sql(f"SHOW VOLUMES IN {CATALOG}.{BRONZE} LIKE '{VOLUME}'").count() > 0
    if not existed:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE}.{VOLUME}")
    return f"{RAW_ROOT} " + ("já existia" if existed else "criado")


def ensure_folder(path):
    def _ensure():
        try:
            dbutils.fs.ls(path)
            return f"já existia: {path}"
        except Exception:
            dbutils.fs.mkdirs(path)
            dbutils.fs.ls(path)  # confirma que ficou criada
            return f"criada: {path}"
    return _ensure


def check_source_files(source):
    def _check():
        present = {f.name for f in dbutils.fs.ls(SOURCES[source])}
        missing = [n for n in EXPECTED_FILES[source] if n not in present]
        if missing:
            raise RuntimeError(f"faltam {missing} em {SOURCES[source]}")
        return f"{len(EXPECTED_FILES[source])} ficheiros presentes"
    return _check


def check_volume_write():
    path = f"{RAW_ROOT}/_setup_check_{uuid.uuid4().hex[:8]}.tmp"
    dbutils.fs.put(path, "ok", True)
    try:
        content = dbutils.fs.head(path)
    finally:
        dbutils.fs.rm(path)
    if content != "ok":
        raise RuntimeError("Leitura do ficheiro de teste não coincide com o escrito")
    return "Escrita, leitura e remoção no Volume OK"


def check_delta_write():
    table = f"{CATALOG}.{BRONZE}._setup_check_{uuid.uuid4().hex[:8]}"
    try:
        spark.range(3).write.format("delta").saveAsTable(table)
        n = spark.table(table).count()
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    if n != 3:
        raise RuntimeError(f"Esperadas 3 linhas, lidas {n}")
    return "Criação, leitura e remoção de tabela Delta OK"


run_check("Sessão Spark", check_spark)
run_check("Catálogo", check_catalog)
for _schema in LAYERS:
    run_check(f"Schema {_schema}", ensure_schema(_schema))
run_check("Volume (bronze.raw)", ensure_volume)
for _source, _path in SOURCES.items():
    run_check(f"Pasta {_source}", ensure_folder(_path))
run_check("Escrita no Volume", check_volume_write)
run_check("Escrita Delta", check_delta_write)
for _source in SOURCES:  # informativo: antes do upload é normal faltarem
    run_check(f"Ficheiros {_source}", check_source_files(_source), critical=False)

# ------------------- 5. Configurações Spark recomendadas ----------------------
# No serverless (Free Edition) muitas configurações são geridas pelo ambiente;
# tenta-se aplicar e regista-se o que ficou efetivo.
SPARK_CONFS = {
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.adaptive.enabled": "true",                     # Adaptive Query Execution
    "spark.sql.adaptive.coalescePartitions.enabled": "true",  # junta partições pequenas
    "spark.sql.adaptive.skewJoin.enabled": "true",            # trata skew em joins
    "spark.sql.autoBroadcastJoinThreshold": "10485760",       # 10 MB
    "spark.databricks.delta.optimizeWrite.enabled": "true",
    "spark.databricks.delta.autoCompact.enabled": "true",
}
conf_report = []
for _key, _wanted in SPARK_CONFS.items():
    try:
        spark.conf.set(_key, _wanted)
        _status = "aplicada"
    except Exception:
        _status = "gerida pelo ambiente"
    try:
        _current = spark.conf.get(_key)
    except Exception:
        _current = "n/d"
    conf_report.append((_key, _wanted, str(_current), _status))
_applied = sum(r[3] == "aplicada" for r in conf_report)
results.append(("Configurações Spark", "OK",
                f"{_applied} aplicadas, {len(conf_report) - _applied} geridas pelo ambiente"))


# --------------- 6. Funções auxiliares (impõem as boas práticas) --------------
def tbl(layer, name):
    """Nome totalmente qualificado: catálogo.camada.tabela."""
    return f"{CATALOG}.{layer}.{name}"


def read_raw_csv(path, columns, sep=",", quote='"', escape="\\", multiline=False):
    """Leitura bronze: schema EXPLÍCITO (sem inferSchema), tudo em string, fiel à origem."""
    schema = ", ".join(f"`{c}` string" for c in columns)
    return (spark.read.schema(schema)
            .option("header", True).option("sep", sep).option("quote", quote)
            .option("escape", escape).option("multiLine", multiline)
            .option("mode", "PERMISSIVE")
            .csv(path))


def with_audit(df, source):
    """Metadados de auditoria da bronze: fonte, ficheiro de origem e momento da ingestão."""
    return (df.withColumn("_source", F.lit(source))
              .withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_ingested_at", F.current_timestamp()))


def nullify(df, tokens=("\\N",), cols=None):
    """Converte tokens de 'em falta' (ex.: \\N do IMDb) em null, num único select (sem withColumn em ciclo)."""
    target = set(cols or [c for c, t in df.dtypes if t == "string"])
    return df.select(*[
        F.when(F.col(c).isin(*tokens), F.lit(None)).otherwise(F.col(c)).alias(c) if c in target else F.col(c)
        for c in df.columns
    ])


def try_cast(col_name, dtype):
    """Cast tolerante: valores inválidos tornam-se null em vez de rebentar (ANSI mode)."""
    return F.expr(f"try_cast(`{col_name}` AS {dtype})")


def safe_date(col_name):
    """Converte 'yyyy-MM-dd' em data; qualquer outro valor (vazio, inválido) fica null."""
    return F.when(F.col(col_name).rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(F.col(col_name)))


def write_delta(df, layer, name, cluster_by=None, comment=None, mode="overwrite"):
    """Escreve uma tabela Delta gerida (Unity Catalog), opcionalmente com Liquid Clustering,
    e devolve o DataFrame lido da tabela (as camadas seguintes partem do que ficou guardado)."""
    full = tbl(layer, name)
    if cluster_by:
        view = f"_w_{uuid.uuid4().hex[:8]}"
        df.createOrReplaceTempView(view)
        spark.sql(f"CREATE OR REPLACE TABLE {full} CLUSTER BY ({', '.join(cluster_by)}) "
                  f"AS SELECT * FROM {view}")
    else:
        writer = df.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(full)
    if comment:
        spark.sql(f"COMMENT ON TABLE {full} IS '{comment.replace(chr(39), chr(39) * 2)}'")
    out = spark.table(full)
    print(f"[{layer}] {full}: {out.count():,} linhas")
    return out


def assert_unique(df, key, label):
    """Controlo de qualidade: a chave tem de ser única (uma só ação Spark)."""
    total, distinct = df.agg(F.count("*"), F.countDistinct(key)).first()
    if total != distinct:
        raise ValueError(f"{label}: chave '{key}' duplicada ({total:,} linhas, {distinct:,} distintas)")
    print(f"[qualidade] {label}: '{key}' única ({total:,} linhas)")


def require_files(source):
    """Falha cedo, com mensagem clara, se faltarem ficheiros da fonte no Volume."""
    present = {f.name for f in dbutils.fs.ls(SOURCES[source])}
    missing = [n for n in EXPECTED_FILES[source] if n not in present]
    if missing:
        raise FileNotFoundError(f"Faltam {missing} em {SOURCES[source]}. Carrega-os (Catalog > Upload to this volume).")


# ----------------------------- 7. Resumo e checklist --------------------------
display(spark.createDataFrame(results, "verificacao string, estado string, detalhe string"))
display(spark.createDataFrame(conf_report, "configuracao string, pretendido string, atual string, estado string"))

BEST_PRACTICES = [
    "Schemas explícitos nas leituras (sem inferSchema); bronze em string, tipagem só na silver",
    "Conversões com try_cast: valores inválidos viram null em vez de rebentar o job",
    "Transformações num único select (sem withColumn em ciclo) e funções nativas (sem UDFs)",
    "Joins: dimensões pequenas com broadcast; filtros e colunas escolhidas o mais cedo possível",
    "Sem collect()/toPandas() em dados grandes; só em tabelas gold já agregadas",
    "Sem cache()/RDDs (não suportados no serverless); cada camada lê a tabela Delta da anterior",
    "Escritas em Delta com overwrite idempotente, comentário na tabela e Liquid Clustering",
    "Controlos de qualidade explícitos (unicidade de chaves, contagens por camada)",
    "Schemas separados por camada (bronze com o Volume de ficheiros brutos, silver, gold) no Unity Catalog",
]
print("\nBoas práticas aplicadas neste notebook:")
for _bp in BEST_PRACTICES:
    print(f"  - {_bp}")

_fails = [r for r in results if r[1] == "FAIL"]
_warns = [r for r in results if r[1] == "WARN"]
print(f"\nOK: {sum(r[1] == 'OK' for r in results)} | WARN: {len(_warns)} | FAIL: {len(_fails)}")
for _name, _, _detail in _warns:
    print(f"  [WARN] {_name}: {_detail}")
if _fails:
    for _name, _, _detail in _fails:
        print(f"  [FAIL] {_name}: {_detail}")
    raise RuntimeError(f"Setup inválido: {len(_fails)} verificação(ões) crítica(s) falharam.")
print("\nSetup válido. Se houver WARN de ficheiros, carrega-os para o Volume antes das pipelines.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pipeline TMDB (JSON)
# MAGIC
# MAGIC **Bronze:** os três ficheiros `.jsonl` guardados tal como vieram (o filme completo continua em JSON dentro de `payload`, a componente não estruturada).
# MAGIC **Silver:** o JSON é interpretado com schema explícito, tipado e deduplicado; os géneros ficam numa tabela própria.

# COMMAND ----------

# =========================== PIPELINE 1/3 · TMDB ===========================
SRC = SOURCES["tmdb"]
require_files("tmdb")

# ------------------------------- BRONZE --------------------------------------
lists_raw = spark.read.schema("list_name string, page int, payload string").json(f"{SRC}/movie_lists.jsonl")
write_delta(with_audit(lists_raw, "tmdb"), BRONZE, "tmdb_movie_lists",
            comment="TMDB: filmes das listas popular/top_rated/now_playing (JSON bruto)")

genres_raw = spark.read.schema("payload string").json(f"{SRC}/genres.jsonl")
write_delta(with_audit(genres_raw, "tmdb"), BRONZE, "tmdb_genres",
            comment="TMDB: lista de géneros (JSON bruto)")

details_raw = spark.read.schema("movie_id long, payload string").json(f"{SRC}/movie_details.jsonl")
write_delta(with_audit(details_raw, "tmdb"), BRONZE, "tmdb_movie_details",
            comment="TMDB: detalhe por filme (JSON bruto)")

# ------------------------------- SILVER --------------------------------------
LIST_MOVIE_DDL = ("id long, title string, original_title string, original_language string, overview string, "
                  "release_date string, popularity double, vote_average double, vote_count long")

tmdb_movies = (
    spark.table(tbl(BRONZE, "tmdb_movie_lists"))
    .select("list_name", F.from_json("payload", LIST_MOVIE_DDL).alias("m"))
    .select("list_name", F.col("m.id").alias("tmdb_id"), "m.title", "m.original_title", "m.original_language",
            "m.overview", F.col("m.release_date").alias("release_date_raw"),
            "m.popularity", "m.vote_average", "m.vote_count")
    .select("list_name", "tmdb_id", "title", "original_title", "original_language", "overview",
            F.when(F.col("release_date_raw").rlike(r"^\d{4}-\d{2}-\d{2}$"),
                   F.to_date("release_date_raw")).alias("release_date"),
            "popularity", "vote_average", "vote_count")
    .groupBy("tmdb_id")
    .agg(F.first("title").alias("title"),
         F.first("original_title").alias("original_title"),
         F.first("original_language").alias("original_language"),
         F.first("overview").alias("overview"),
         F.first("release_date").alias("release_date"),
         F.max("popularity").alias("popularity"),
         F.max("vote_average").alias("vote_average"),
         F.max("vote_count").alias("vote_count"),
         F.collect_set("list_name").alias("lists"))
    .withColumn("release_year", F.year("release_date"))
)
assert_unique(tmdb_movies, "tmdb_id", "silver.tmdb_movies")
write_delta(tmdb_movies, SILVER, "tmdb_movies",
            comment="TMDB: filmes das listas, um por tmdb_id, com as listas em que aparecem")

DETAIL_DDL = ("id long, imdb_id string, title string, original_title string, original_language string, "
              "overview string, release_date string, runtime int, budget long, revenue long, status string, "
              "popularity double, vote_average double, vote_count long, "
              "genres array<struct<id:int,name:string>>")

details_parsed = (
    spark.table(tbl(BRONZE, "tmdb_movie_details"))
    .select(F.from_json("payload", DETAIL_DDL).alias("d"))
    .select("d.*")
)
tmdb_details = (
    nullify(details_parsed, tokens=("",))
    .select(F.col("id").alias("tmdb_id"), "imdb_id", "title", "original_title", "original_language", "overview",
            safe_date("release_date").alias("release_date"),
            F.col("runtime").alias("runtime_minutes"), "budget", "revenue", "status",
            "popularity", "vote_average", "vote_count", "genres")
    .dropDuplicates(["tmdb_id"])
)
assert_unique(tmdb_details, "tmdb_id", "silver.tmdb_movie_details")
tmdb_details = write_delta(tmdb_details, SILVER, "tmdb_movie_details",
                           comment="TMDB: detalhe tipado (orçamento, receita, duração, géneros)")

tmdb_movie_genres = (tmdb_details
                     .select("tmdb_id", F.explode("genres").alias("g"))
                     .select("tmdb_id", F.col("g.id").alias("genre_id"), F.col("g.name").alias("genre_name")))
write_delta(tmdb_movie_genres, SILVER, "tmdb_movie_genres",
            comment="TMDB: relação filme-género (nomes no idioma pedido ao descarregar)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pipeline IMDb (TSV)
# MAGIC
# MAGIC **Bronze:** `title.basics` e `title.ratings` (já filtrados para `titleType = movie` no download), tudo em string, com `\N` preservado.
# MAGIC **Silver:** `\N` passa a `null`, colunas tipadas, adultos removidos, géneros em array e nota/votos juntos ao filme. Tabela com **Liquid Clustering** por `start_year`.

# COMMAND ----------

# ============================ PIPELINE 2/3 · IMDb ============================
SRC = SOURCES["imdb"]
require_files("imdb")

BASICS_COLS  = ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                "startYear", "endYear", "runtimeMinutes", "genres"]
RATINGS_COLS = ["tconst", "averageRating", "numVotes"]

# ------------------------------- BRONZE --------------------------------------
# TSV do IMDb: separador tab e sem aspas (os títulos podem conter aspas).
basics_raw  = read_raw_csv(f"{SRC}/title.basics.tsv.gz",  BASICS_COLS,  sep="\t", quote="")
ratings_raw = read_raw_csv(f"{SRC}/title.ratings.tsv.gz", RATINGS_COLS, sep="\t", quote="")

write_delta(with_audit(basics_raw, "imdb"), BRONZE, "imdb_title_basics",
            comment="IMDb title.basics (só filmes), valores brutos em string")
write_delta(with_audit(ratings_raw, "imdb"), BRONZE, "imdb_title_ratings",
            comment="IMDb title.ratings (só filmes), valores brutos em string")

# ------------------------------- SILVER --------------------------------------
basics  = nullify(spark.table(tbl(BRONZE, "imdb_title_basics")))
ratings = nullify(spark.table(tbl(BRONZE, "imdb_title_ratings")))

imdb_ratings = ratings.select("tconst",
                              try_cast("averageRating", "double").alias("imdb_rating"),
                              try_cast("numVotes", "bigint").alias("imdb_votes"))

imdb_movies = (
    basics.where(F.col("titleType") == "movie")
    .select("tconst",
            F.col("primaryTitle").alias("title"),
            F.col("originalTitle").alias("original_title"),
            (F.col("isAdult") == "1").alias("is_adult"),
            try_cast("startYear", "int").alias("start_year"),
            try_cast("runtimeMinutes", "int").alias("runtime_minutes"),
            F.split("genres", ",").alias("genres"))
    .where(~F.coalesce(F.col("is_adult"), F.lit(False)))
    .dropDuplicates(["tconst"])
    .join(imdb_ratings, "tconst", "left")
)
assert_unique(imdb_movies, "tconst", "silver.imdb_movies")
imdb_movies = write_delta(imdb_movies, SILVER, "imdb_movies", cluster_by=["start_year"],
                          comment="IMDb: filmes não adultos, tipados, com nota e votos (Liquid Clustering por start_year)")

imdb_movie_genres = imdb_movies.select("tconst", F.explode("genres").alias("genre"))
write_delta(imdb_movie_genres, SILVER, "imdb_movie_genres",
            comment="IMDb: relação filme-género (nomes em inglês)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pipeline MovieLens (CSV)
# MAGIC
# MAGIC **Bronze:** `ratings`, `movies`, `links` e `tags` como vieram (CSV com aspas duplicadas, daí `escape='"'`).
# MAGIC **Silver:** avaliações com timestamp real, título separado do ano, géneros em array, `tconst` construído a partir do `imdbId` e tags normalizadas (texto livre).

# COMMAND ----------

# ========================= PIPELINE 3/3 · MovieLens ==========================
SRC = SOURCES["movielens"]
require_files("movielens")

# ------------------------------- BRONZE --------------------------------------
ML_SCHEMAS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies":  ["movieId", "title", "genres"],
    "links":   ["movieId", "imdbId", "tmdbId"],
    "tags":    ["userId", "movieId", "tag", "timestamp"],
}
for name, cols in ML_SCHEMAS.items():
    raw = read_raw_csv(f"{SRC}/{name}.csv", cols, escape='"', multiline=(name != "ratings"))
    write_delta(with_audit(raw, "movielens"), BRONZE, f"ml_{name}",
                comment=f"MovieLens {name}.csv, valores brutos em string")

# ------------------------------- SILVER --------------------------------------
ml_ratings = (
    spark.table(tbl(BRONZE, "ml_ratings"))
    .select(try_cast("userId", "int").alias("user_id"),
            try_cast("movieId", "int").alias("movie_id"),
            try_cast("rating", "double").alias("rating"),
            F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("rated_at"))
    .where(F.col("movie_id").isNotNull() & F.col("rating").isNotNull())
    .dropDuplicates(["user_id", "movie_id", "rated_at"])
)
write_delta(ml_ratings, SILVER, "ml_ratings", cluster_by=["movie_id"],
            comment="MovieLens: avaliações tipadas (escala 0.5 a 5)")

YEAR_PATTERN = r"\((\d{4})\)\s*$"
ml_movies = (
    spark.table(tbl(BRONZE, "ml_movies"))
    .select(try_cast("movieId", "int").alias("movie_id"),
            F.trim(F.regexp_replace("title", YEAR_PATTERN, "")).alias("title"),
            F.when(F.regexp_extract("title", YEAR_PATTERN, 1) != "",
                   F.regexp_extract("title", YEAR_PATTERN, 1).cast("int")).alias("year"),
            F.filter(F.split("genres", r"\|"),
                     lambda g: (g != "") & (g != "(no genres listed)")).alias("genres"))
    .dropDuplicates(["movie_id"])
)
assert_unique(ml_movies, "movie_id", "silver.ml_movies")
write_delta(ml_movies, SILVER, "ml_movies", comment="MovieLens: filmes com título, ano e géneros em array")

ml_links = (
    spark.table(tbl(BRONZE, "ml_links"))
    .select(try_cast("movieId", "int").alias("movie_id"),
            F.concat(F.lit("tt"), F.lpad("imdbId", 7, "0")).alias("tconst"),   # chave do IMDb
            try_cast("tmdbId", "int").alias("tmdb_id"))                        # chave do TMDB
    .dropDuplicates(["movie_id"])
)
assert_unique(ml_links, "movie_id", "silver.ml_links")
write_delta(ml_links, SILVER, "ml_links",
            comment="MovieLens: ponte movie_id <-> tconst (IMDb) <-> tmdb_id (TMDB)")

ml_tags = (
    spark.table(tbl(BRONZE, "ml_tags"))
    .select(try_cast("userId", "int").alias("user_id"),
            try_cast("movieId", "int").alias("movie_id"),
            F.lower(F.trim("tag")).alias("tag"),
            F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("tagged_at"))
    .where(F.col("tag").isNotNull() & (F.col("tag") != ""))
)
write_delta(ml_tags, SILVER, "ml_tags", comment="MovieLens: tags de texto livre normalizadas (minúsculas)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Silver integrada: `silver.movies_unified`
# MAGIC
# MAGIC Junta as três fontes por chave. O **IMDb é a tabela grande** e as restantes são pequenas, por isso usam-se **Broadcast Hash Joins**. O plano de execução (`explain`) mostra isso mesmo.
# MAGIC Valores `0` de orçamento, receita e votos do TMDB significam "desconhecido", por isso passam a `null`.

# COMMAND ----------

# =========================== SILVER INTEGRADA ================================
imdb  = spark.table(tbl(SILVER, "imdb_movies"))
links = spark.table(tbl(SILVER, "ml_links"))
tmdb  = spark.table(tbl(SILVER, "tmdb_movie_details"))

ml_stats = (spark.table(tbl(SILVER, "ml_ratings"))
            .groupBy("movie_id")
            .agg(F.round(F.avg("rating"), 3).alias("ml_avg_rating"),
                 F.count("*").alias("ml_num_ratings")))

tmdb_sel = tmdb.select(
    "tmdb_id", "original_language", "overview",
    F.when(F.col("vote_count") > 0, F.col("vote_average")).alias("tmdb_vote_average"),
    F.col("vote_count").alias("tmdb_vote_count"),
    F.col("popularity").alias("tmdb_popularity"),
    F.when(F.col("budget") > 0, F.col("budget")).alias("budget"),
    F.when(F.col("revenue") > 0, F.col("revenue")).alias("revenue"),
    F.lit(True).alias("in_tmdb"),
)

unified = (
    imdb.join(F.broadcast(links), "tconst", "inner")        # IMDb (grande) x ponte MovieLens (pequena)
        .join(F.broadcast(tmdb_sel), "tmdb_id", "left")     # + TMDB
        .join(F.broadcast(ml_stats), "movie_id", "left")    # + avaliações MovieLens
        .select("movie_id", "tconst", "tmdb_id", "title", "start_year",
                (F.floor(F.col("start_year") / 10) * 10).alias("decade"),
                "runtime_minutes", "genres", "imdb_rating", "imdb_votes",
                "ml_avg_rating", (F.col("ml_avg_rating") * 2).alias("ml_rating_10"), "ml_num_ratings",
                "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity",
                "budget", "revenue", "original_language", "overview",
                F.coalesce(F.col("in_tmdb"), F.lit(False)).alias("in_tmdb"))
)

unified.explain(mode="formatted")  # evidência do BroadcastHashJoin para o relatório

assert_unique(unified, "movie_id", "silver.movies_unified")
unified = write_delta(unified, SILVER, "movies_unified", cluster_by=["decade", "start_year"],
                      comment="Filmes integrados (IMDb + MovieLens + TMDB); ml_rating_10 = nota MovieLens na escala 0-10")

# Cobertura entre fontes
total, with_tmdb, with_ml = unified.agg(
    F.count("*"),
    F.sum(F.col("in_tmdb").cast("int")),
    F.sum(F.col("ml_num_ratings").isNotNull().cast("int"))).first()
print(f"Filmes integrados: {total:,} | com TMDB: {with_tmdb:,} | com avaliações MovieLens: {with_ml:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gold: tabelas para responder à pergunta de negócio
# MAGIC
# MAGIC Para evitar ruído usam-se limiares mínimos (`MIN_IMDB_VOTES`, `MIN_ML_RATINGS`). A nota do MovieLens é multiplicada por 2 (`ml_rating_10`) para ficar na mesma escala 0-10 do IMDb e do TMDB.
# MAGIC Os géneros usados são os do IMDb (inglês), porque os do TMDB vêm no idioma pedido (pt-PT).

# COMMAND ----------

# ================================== GOLD =====================================
MIN_IMDB_VOTES = 1000
MIN_ML_RATINGS = 5

base = (spark.table(tbl(SILVER, "movies_unified"))
        .where((F.col("imdb_votes") >= MIN_IMDB_VOTES)
               & (F.col("ml_num_ratings") >= MIN_ML_RATINGS)
               & F.col("decade").isNotNull()))

rating_metrics = [
    F.count("*").alias("n_movies"),
    F.round(F.avg("imdb_rating"), 2).alias("avg_imdb_rating"),
    F.round(F.avg("ml_rating_10"), 2).alias("avg_movielens_rating_10"),
    F.round(F.avg("tmdb_vote_average"), 2).alias("avg_tmdb_rating"),
]

# 1) Nota média por fonte e por década
by_decade = base.groupBy("decade").agg(*rating_metrics, F.sum(F.col("in_tmdb").cast("int")).alias("n_with_tmdb"))
write_delta(by_decade, GOLD, "ratings_by_decade", comment="Nota média por fonte e década")

# 2) Nota média por fonte e por género (género do IMDb)
by_genre = (base.withColumn("genre", F.explode("genres"))
            .groupBy("genre").agg(*rating_metrics)
            .where(F.col("n_movies") >= 10))
write_delta(by_genre, GOLD, "ratings_by_genre", comment="Nota média por fonte e género (IMDb, mín. 10 filmes)")

# 3) Filmes onde o IMDb e o MovieLens mais divergem
gaps = (base.select("movie_id", "tconst", "title", "start_year", "imdb_rating", "imdb_votes",
                    "ml_rating_10", "ml_num_ratings",
                    F.round(F.col("imdb_rating") - F.col("ml_rating_10"), 2).alias("gap_imdb_minus_ml"))
        .withColumn("abs_gap", F.abs("gap_imdb_minus_ml")))
write_delta(gaps, GOLD, "movie_rating_gaps", cluster_by=["start_year"],
            comment="Diferença entre a nota IMDb e a nota MovieLens (escala 0-10) por filme")

# 4) Retorno financeiro por género (só filmes com orçamento e receita conhecidos no TMDB)
roi = (spark.table(tbl(SILVER, "movies_unified"))
       .where(F.col("budget").isNotNull() & F.col("revenue").isNotNull())
       .withColumn("genre", F.explode("genres"))
       .withColumn("roi", F.col("revenue") / F.col("budget"))
       .groupBy("genre")
       .agg(F.count("*").alias("n_movies"),
            F.round(F.avg("roi"), 2).alias("avg_roi"),
            F.round(F.percentile_approx("roi", 0.5), 2).alias("median_roi"),
            F.round(F.avg("imdb_rating"), 2).alias("avg_imdb_rating"))
       .where(F.col("n_movies") >= 5))
write_delta(roi, GOLD, "roi_by_genre", comment="Retorno (receita/orçamento) por género, filmes com TMDB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Análise e visualização

# COMMAND ----------

display(spark.table(tbl(GOLD, "ratings_by_decade")).orderBy("decade"))

# COMMAND ----------

display(spark.table(tbl(GOLD, "ratings_by_genre")).orderBy(F.desc("n_movies")))

# COMMAND ----------

# Os 20 filmes com maior divergência entre IMDb e MovieLens
display(spark.table(tbl(GOLD, "movie_rating_gaps")).orderBy(F.desc("abs_gap")).limit(20))

# COMMAND ----------

display(spark.table(tbl(GOLD, "roi_by_genre")).orderBy(F.desc("median_roi")))

# COMMAND ----------

import matplotlib.pyplot as plt

# toPandas() só é aceitável aqui: tabela gold já agregada (poucas linhas)
pdf = spark.table(tbl(GOLD, "ratings_by_decade")).orderBy("decade").toPandas()

fig, ax = plt.subplots(figsize=(9, 4.5))
for col, label in [("avg_imdb_rating", "IMDb"),
                   ("avg_movielens_rating_10", "MovieLens (escala 0-10)"),
                   ("avg_tmdb_rating", "TMDB")]:
    ax.plot(pdf["decade"], pdf[col], marker="o", label=label)
ax.set_title("Nota média por década e fonte")
ax.set_xlabel("Década")
ax.set_ylabel("Nota média (0-10)")
ax.grid(alpha=0.3)
ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Otimização e evidências para o relatório
# MAGIC
# MAGIC Comentar em Markdown, com base nos resultados abaixo: lazy execution e lineage (`explain`), Broadcast Hash Join, Adaptive Query Execution, Liquid Clustering e Delta Lake (`DESCRIBE HISTORY`).
# MAGIC _Opcional:_ comparar o tamanho e o tempo de leitura de CSV, Parquet e Delta para a mesma tabela.

# COMMAND ----------

# Plano de execução de uma consulta gold (procurar BroadcastHashJoin / AdaptiveSparkPlan)
(spark.table(tbl(SILVER, "movies_unified"))
      .join(F.broadcast(spark.table(tbl(SILVER, "ml_movies")).select("movie_id", F.col("year").alias("ml_year"))),
            "movie_id")
      .where(F.col("decade") == 1990)
      .groupBy("decade").agg(F.avg("imdb_rating").alias("avg_imdb_rating"))
      .explain(mode="formatted"))

# COMMAND ----------

# Liquid Clustering: ver as colunas de clustering e o histórico (time travel) da tabela
display(spark.sql(f"DESCRIBE DETAIL {tbl(SILVER, 'movies_unified')}")
        .select("name", "format", "numFiles", "sizeInBytes", "clusteringColumns"))
display(spark.sql(f"DESCRIBE HISTORY {tbl(SILVER, 'movies_unified')}")
        .select("version", "timestamp", "operation"))

# COMMAND ----------

# Inventário final: tabelas e linhas por camada
inventory = []
for layer in LAYERS:
    for row in spark.sql(f"SHOW TABLES IN {CATALOG}.{layer}").collect():
        if not row.isTemporary:
            n = spark.table(tbl(layer, row.tableName)).count()
            inventory.append((layer, row.tableName, n))
display(spark.createDataFrame(inventory, "camada string, tabela string, linhas long").orderBy("camada", "tabela"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conclusões
# MAGIC
# MAGIC _(Responder à pergunta de negócio com base nas tabelas gold: diferenças de nota entre fontes, efeito do género e da década, retorno financeiro.)_
# MAGIC
# MAGIC **Limitações:** o MovieLens-small cobre ~9 mil filmes; o TMDB só tem detalhe para os filmes descarregados; os géneros do TMDB estão em português; os dados foram descarregados localmente por limitação da Free Edition.
# MAGIC
# MAGIC **Trabalho futuro:** usar um MovieLens maior, acrescentar `title.principals` do IMDb (elenco/realizadores) e analisar o texto das sinopses e tags.