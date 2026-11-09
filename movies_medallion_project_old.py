# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Projeto de grupo: as notas dos filmes dependem de quem as dá?
# MAGIC ### IMDb + TMDB + MovieLens num Data Lakehouse (Arquitetura Medallion) seguindo o CRISP-DM
# MAGIC
# MAGIC **Grupo:** _(nome do grupo)_
# MAGIC **Elementos:** _(Francisco Monteiro, nº 20251694)_ · _(José Grave, nº 20251843)_ · _(Luís Pousada, nº)_ · _(Tiago Felgueira, nº20242047)_
# MAGIC **Curso:** Analyzing Big Data 2026, NOVA IMS  |  **Data de entrega:** 16/11/2026
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Organização do notebook
# MAGIC
# MAGIC Usamos duas ideias que se complementam:
# MAGIC
# MAGIC * a **arquitetura Medallion** organiza os **dados** (bronze = bruto com histórico, silver = limpo e integrado, gold = pronto para análise);
# MAGIC * o **CRISP-DM** organiza o **processo de análise** (perceber o problema, perceber os dados, preparar, modelar, avaliar, pôr em produção).
# MAGIC
# MAGIC | Fase CRISP-DM | Secção | Camada Medallion |
# MAGIC |---|---|---|
# MAGIC | 1. Business Understanding | 1 | — |
# MAGIC | 2. Data Understanding | 2 | Volume `raw` + **Bronze** |
# MAGIC | 3. Data Preparation | 3 | **Silver** |
# MAGIC | 4. Modeling | 4 | **Gold** |
# MAGIC | 5. Evaluation | 5 | Gold + gráficos |
# MAGIC | 6. Deployment | 6 | Job do Databricks (pipeline visual) |
# MAGIC
# MAGIC ### Como executar
# MAGIC
# MAGIC * **Tudo de uma vez:** deixar o widget `etapa = tudo` e fazer **Run all**.
# MAGIC * **Só uma parte:** escolher a etapa no widget (`tmdb`, `imdb`, `movielens`, `integracao`, `gold`, `analise`) e fazer **Run all**. As células das outras etapas são saltadas.
# MAGIC * **Como pipeline visual:** criar o Job descrito na secção 6. Cada tarefa do Job corre este mesmo notebook com uma etapa diferente, e o Databricks desenha o grafo das tarefas.
# MAGIC
# MAGIC ### Guia de leitura
# MAGIC
# MAGIC Nem todas as células têm o mesmo peso. Para perceber o projeto basta ler as células **principais**; as **auxiliares** existem para o notebook ser seguro e apresentável, e começam sempre por um comentário `[Auxiliar]`.
# MAGIC
# MAGIC | Tipo | O que fazem | Onde estão |
# MAGIC |---|---|---|
# MAGIC | **Principal** | lê os ficheiros, limpa e junta os dados, calcula as tabelas gold e desenha os gráficos (o "Spark" do projeto) | bronze (2), silver (3), gold (4, 4.3), gráficos (5) |
# MAGIC | **Auxiliar** | verificações no arranque, migração de tabelas antigas, cópias de segurança, constraints, diagramas em HTML/SVG | setup (0), 1.1, 4.2 (desenho e constraints), inventário final |
# MAGIC
# MAGIC Cada função tem um nome que diz o que faz (`bronze_imdb`, `silver_tmdb`, `gold_top_movies`...) e é chamada logo a seguir com `if correr("etapa"):`, para se poder correr só uma parte do pipeline.
# MAGIC
# MAGIC **Padrões de PySpark que aparecem muitas vezes:**
# MAGIC * `F.col("x")` refere uma coluna; `withColumn` cria ou substitui uma coluna; `groupBy(...).agg(...)` agrega; `join` junta tabelas.
# MAGIC * `Window.partitionBy(...).orderBy(...)` faz contas "por grupo sem agrupar".
# MAGIC * `lambda p: ...` é uma função pequena sem nome; usamo-la para dizer *como* se lê um ficheiro, deixando a outra função decidir *se* o lê.
# MAGIC * Nada é calculado até haver uma ação (`count`, `write`, `display`): o Spark é *lazy* e só executa no fim.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup
# MAGIC
# MAGIC Esta célula corre sempre primeiro. Lê os parâmetros (widgets), confirma o ambiente, define as funções auxiliares e **verifica todos os objetos que já existem** antes de mexer em qualquer um.
# MAGIC
# MAGIC **Regras para não perder dados nem ter erros ao voltar a correr**
# MAGIC
# MAGIC | Objeto | Se não existe | Se já existe |
# MAGIC |---|---|---|
# MAGIC | Catálogo | erro com instruções (não é criado pelo notebook) | usado |
# MAGIC | Schemas `bronze`, `silver`, `gold` e Volume `raw` | criados | mantidos, nunca apagados |
# MAGIC | Ficheiros no Volume | erro só se a tabela bronze também não existir | lidos; nunca apagados nem alterados |
# MAGIC | Tabelas **bronze** | criadas na 1.ª ingestão | **só append** de ficheiros novos; tabelas de versões antigas são **migradas** (colunas novas, linhas mantidas); histórico Delta guardado 90 dias |
# MAGIC | Tabelas **silver** e **gold** | criadas | reconstruídas a partir da camada anterior (escrita atómica; a versão anterior fica no histórico Delta) |
# MAGIC | `gold.dq_results` | criada | só append (histórico da qualidade) |
# MAGIC
# MAGIC O widget `reiniciar` tem três opções:
# MAGIC * `nao` (por omissão): não apaga nada;
# MAGIC * `silver_gold`: apaga só silver e gold, que são reconstruídas a partir da bronze, sem perda de informação;
# MAGIC * `tudo_com_backup`: copia primeiro as tabelas bronze e `dq_results` para o schema `backup` e só depois as apaga. Uma fonte sem ficheiros no Volume nunca é apagada.
# MAGIC
# MAGIC O reinício só é aceite com `etapa = tudo`, nunca numa tarefa de um Job.
# MAGIC
# MAGIC Se algo correr mal, qualquer tabela Delta pode voltar atrás: `DESCRIBE HISTORY tabela` para ver as versões e `RESTORE TABLE tabela TO VERSION AS OF n`.

# COMMAND ----------

import math
import uuid

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

# ------------------------------- 1. Parâmetros --------------------------------
ETAPAS = ["tudo", "tmdb", "imdb", "movielens", "integracao", "gold", "analise"]

dbutils.widgets.text("catalog", "abd_project", "Catálogo")
dbutils.widgets.dropdown("etapa", "tudo", ETAPAS, "Etapa a executar")
dbutils.widgets.dropdown("reiniciar", "nao", ["nao", "silver_gold", "tudo_com_backup"], "Reiniciar tabelas")
dbutils.widgets.text("min_imdb_votes", "1000", "Mín. votos IMDb")
dbutils.widgets.text("min_ml_ratings", "10", "Mín. avaliações MovieLens")
dbutils.widgets.text("min_tmdb_votes", "20", "Mín. votos TMDB")

CATALOG        = dbutils.widgets.get("catalog").strip()
ETAPA          = dbutils.widgets.get("etapa")
REINICIAR      = dbutils.widgets.get("reiniciar")   # nao | silver_gold | tudo_com_backup


def _int_widget(name, default):
    """Lê um widget numérico; se o valor for inválido usa o valor por omissão em vez de dar erro."""
    try:
        return max(0, int(dbutils.widgets.get(name)))
    except ValueError:
        print(f"[aviso] widget '{name}' inválido; a usar {default}")
        return default


MIN_IMDB_VOTES = _int_widget("min_imdb_votes", 1000)
MIN_ML_RATINGS = _int_widget("min_ml_ratings", 10)
MIN_TMDB_VOTES = _int_widget("min_tmdb_votes", 20)

# Um reinício só pode ser feito numa execução completa e manual, nunca numa tarefa de um Job
# (apagaria tabelas que outras tarefas estão a ler ao mesmo tempo).
if REINICIAR != "nao" and ETAPA != "tudo":
    raise ValueError("O reinício só é permitido com etapa = tudo. Num Job usa sempre reiniciar = nao.")


def correr(*etapas):
    """True se a etapa escolhida no widget for 'tudo' ou uma das etapas indicadas."""
    return ETAPA == "tudo" or ETAPA in etapas


BRONZE, SILVER, GOLD = "bronze", "silver", "gold"
LAYERS   = [BRONZE, SILVER, GOLD]
VOLUME   = "raw"
RAW_ROOT = f"/Volumes/{CATALOG}/{BRONZE}/{VOLUME}"
SOURCES  = {s: f"{RAW_ROOT}/{s}" for s in ("tmdb", "imdb", "movielens")}

# Nomes dos ficheiros num só sítio: usados tanto na verificação como na leitura.
# (Ajustar aqui se os ficheiros tiverem outro nome no Volume.)
FILES = {
    "tmdb":      {"lists": "movie_lists.jsonl", "genres": "genres.jsonl", "details": "movie_details.jsonl"},
    "imdb":      {"basics": "title.basics.movies.tsv.gz", "ratings": "title.ratings.movies.tsv.gz"},
    "movielens": {"ratings": "ratings.csv", "movies": "movies.csv", "links": "links.csv", "tags": "tags.csv"},
}

# Identificador desta execução: cada leitura para a bronze fica marcada com ele.
BATCH_ID = spark.sql("SELECT date_format(current_timestamp(), 'yyyyMMdd_HHmmss')").first()[0]

# Tabelas de cada camada (usadas no diagrama do pipeline e no inventário)
BRONZE_TABLES = {"tmdb": ["tmdb_movie_lists", "tmdb_genres", "tmdb_movie_details"],
                 "imdb": ["imdb_title_basics", "imdb_title_ratings"],
                 "movielens": ["ml_ratings", "ml_movies", "ml_links", "ml_tags"]}
SILVER_TABLES = {"tmdb": ["tmdb_movies", "tmdb_movie_details", "tmdb_movie_genres"],
                 "imdb": ["imdb_movies", "imdb_movie_genres"],
                 "movielens": ["ml_ratings", "ml_movies", "ml_links", "ml_tags"]}
GOLD_TABLES = ["ratings_by_decade", "gap_by_genre", "gap_by_language", "gap_by_budget",
               "movie_rating_gaps", "roi_by_genre", "tags_by_gap", "gap_model_coefficients", "dq_results",
               # perguntas descritivas e de benchmarking (secção 4.3)
               "genre_catalog", "user_activity", "top_movies", "ratings_over_time", "flag_sensitivity",
               "movie_rating_shifts", "movie_day_anomalies", "anomaly_participants"]
# Modelo dimensional (star schema) na gold. A tabela de factos vem primeiro: num reinício é apagada
# antes das dimensões a que faz referência.
STAR_TABLES = ["fact_rating", "bridge_movie_genre", "dim_movie", "dim_source", "dim_year", "dim_language", "dim_genre"]
LANG_NAMES = {"en": "Inglês", "fr": "Francês", "it": "Italiano", "ja": "Japonês", "es": "Espanhol",
              "de": "Alemão", "ko": "Coreano", "zh": "Mandarim", "cn": "Cantonês", "hi": "Hindi",
              "sv": "Sueco", "ru": "Russo", "da": "Dinamarquês", "pt": "Português", "nl": "Neerlandês"}

# ------------------------- 2. Verificações do ambiente ------------------------
# [Auxiliar] Confirma catálogo, schemas, Volume e ficheiros antes de começar. Não é preciso
# perceber cada função: basta ler a tabela de verificações que aparece no output.
results = []  # (verificação, estado, detalhe)


def run_check(name, fn, critical=True):
    """Executa uma verificação e regista OK, FAIL (crítica) ou WARN (não crítica)."""
    try:
        results.append((name, "OK", str(fn() or "")))
    except Exception as e:
        results.append((name, "FAIL" if critical else "WARN", f"{type(e).__name__}: {e}"))


def check_catalog():
    catalogs = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
    if CATALOG not in catalogs:
        raise RuntimeError(f"Catálogo '{CATALOG}' não existe. Cria-o em Catalog > + > Create catalog, "
                           f"ou escolhe outro no widget. Disponíveis: {catalogs}")
    return f"Catálogo '{CATALOG}' já existia"


def schema_exists(schema):
    return spark.sql(f"SHOW SCHEMAS IN {CATALOG} LIKE '{schema}'").count() > 0


def ensure_schema(schema):
    """Cria o schema só se não existir. Nunca apaga nada."""
    def _ensure():
        if schema_exists(schema):
            return f"{CATALOG}.{schema} já existia (mantido)"
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
        except Exception:
            if not schema_exists(schema):   # outra tarefa do Job pode tê-lo criado ao mesmo tempo
                raise
        return f"{CATALOG}.{schema} criado"
    return _ensure


def volume_exists():
    return spark.sql(f"SHOW VOLUMES IN {CATALOG}.{BRONZE} LIKE '{VOLUME}'").count() > 0


def ensure_volume():
    """Cria o Volume e as pastas das fontes só se não existirem. Os ficheiros nunca são apagados."""
    existed = volume_exists()
    if not existed:
        try:
            spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE}.{VOLUME}")
        except Exception:
            if not volume_exists():
                raise
    for path in SOURCES.values():
        dbutils.fs.mkdirs(path)  # não faz nada se a pasta já existir
    return f"{RAW_ROOT} " + ("já existia (mantido)" if existed else "criado") + "; pastas das fontes prontas"


# Um ficheiro "conta" se começar pelo mesmo nome base: movie_details.jsonl, movie_details_2026-11-10.jsonl, ...
EXTENSIONS = (".tsv.gz", ".jsonl", ".csv")


def file_pattern(source, key):
    """(nome base, extensão) do ficheiro esperado, ex.: ('movie_details', '.jsonl')."""
    fname = FILES[source][key]
    ext = next(e for e in EXTENSIONS if fname.endswith(e))
    return fname[: -len(ext)], ext


def source_glob(source, key):
    stem, ext = file_pattern(source, key)
    return f"{SOURCES[source]}/{stem}*{ext}"


def source_files(source, key):
    """Ficheiros do Volume que correspondem a FILES[source][key] (lista vazia se não houver)."""
    stem, ext = file_pattern(source, key)
    try:
        return [f.name for f in dbutils.fs.ls(SOURCES[source]) if f.name.startswith(stem) and f.name.endswith(ext)]
    except Exception:
        return []


def missing_files(source):
    return [FILES[source][k] for k in FILES[source] if not source_files(source, k)]


def check_source_files(source):
    def _check():
        missing = missing_files(source)
        if missing:
            raise RuntimeError(f"faltam {missing} em {SOURCES[source]}")
        n = sum(len(source_files(source, k)) for k in FILES[source])
        return f"{n} ficheiro(s) encontrados"
    return _check


run_check("Sessão Spark", lambda: f"Spark {spark.version}")
run_check("Catálogo", check_catalog)
for _schema in LAYERS:
    run_check(f"Schema {_schema}", ensure_schema(_schema))
run_check("Volume bronze.raw", ensure_volume)
for _source in SOURCES:  # não crítico: se a bronze já tiver histórico, a falta de ficheiros não impede nada
    run_check(f"Ficheiros {_source}", check_source_files(_source), critical=False)

# Configurações recomendadas (no serverless algumas são geridas pelo ambiente)
for _key, _value in {"spark.sql.session.timeZone": "UTC",
                     "spark.sql.adaptive.enabled": "true",
                     "spark.sql.adaptive.skewJoin.enabled": "true"}.items():
    try:
        spark.conf.set(_key, _value)
    except Exception:
        pass

# ------------------------- 3. Funções auxiliares ------------------------------
def tbl(layer, name):
    """Nome completo da tabela: catálogo.camada.tabela."""
    return f"{CATALOG}.{layer}.{name}"


def read_raw_csv(path, columns, sep=",", quote='"', escape="\\", multiline=False):
    """Leitura para a bronze: schema explícito, tudo em string (fiel à origem)."""
    schema = ", ".join(f"`{c}` string" for c in columns)
    return (spark.read.schema(schema)
            .option("header", True).option("sep", sep).option("quote", quote)
            .option("escape", escape).option("multiLine", multiline)
            .csv(path))


def with_audit(df, source):
    """Metadados de auditoria: fonte, ficheiro, data do ficheiro, lote e momento da ingestão."""
    return (df.withColumn("_source", F.lit(source))
              .withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_file_modified", F.col("_metadata.file_modification_time"))
              .withColumn("_batch_id", F.lit(BATCH_ID))
              .withColumn("_ingested_at", F.current_timestamp()))


def set_comment(full_name, comment):
    """Guarda uma descrição da tabela no Unity Catalog (aparece no Catalog Explorer)."""
    if comment:
        texto = comment.replace("'", "''")   # em SQL, uma aspa dentro de um texto escreve-se duas vezes
        spark.sql(f"COMMENT ON TABLE {full_name} IS '{texto}'")


# Colunas de auditoria que todas as tabelas bronze devem ter
BRONZE_AUDIT_COLS = {"_source": "STRING", "_source_file": "STRING", "_file_modified": "TIMESTAMP",
                     "_batch_id": "STRING", "_ingested_at": "TIMESTAMP"}
# Histórico Delta guardado mais tempo na bronze: permite recuperar versões antigas (time travel / RESTORE)
BRONZE_PROPERTIES = {"delta.logRetentionDuration": "interval 90 days",
                     "delta.deletedFileRetentionDuration": "interval 90 days"}


# [Auxiliar] As duas funções seguintes tratam da "manutenção" da bronze: guardar o histórico
# Delta durante 90 dias e atualizar tabelas criadas por versões antigas do notebook.
def protect_bronze(full):
    props = ", ".join(f"'{k}' = '{v}'" for k, v in BRONZE_PROPERTIES.items())
    try:
        spark.sql(f"ALTER TABLE {full} SET TBLPROPERTIES ({props})")
    except Exception as e:  # não crítico: algumas propriedades podem ser geridas pelo ambiente
        print(f"[aviso] {full}: não foi possível definir a retenção do histórico ({type(e).__name__})")
    # Organização física: Liquid Clustering pelo lote de ingestão. Não reescreve dados já existentes
    # (só se aplica às escritas seguintes e ao OPTIMIZE), por isso é seguro em tabelas com histórico.
    try:
        spark.sql(f"ALTER TABLE {full} CLUSTER BY (_batch_id)")
    except Exception as e:
        print(f"[aviso] {full}: Liquid Clustering não disponível neste ambiente ({type(e).__name__})")


def migrate_bronze(full):
    """Atualiza uma tabela bronze criada por uma versão antiga do notebook, SEM apagar linhas:
    acrescenta as colunas de auditoria em falta e marca as linhas antigas com o lote 'legado'."""
    missing = {c: t for c, t in BRONZE_AUDIT_COLS.items() if c not in spark.table(full).columns}
    if not missing:
        return False
    spark.sql(f"ALTER TABLE {full} ADD COLUMNS ({', '.join(f'{c} {t}' for c, t in missing.items())})")
    if "_batch_id" in missing:
        spark.sql(f"UPDATE {full} SET _batch_id = 'legado' WHERE _batch_id IS NULL")
    protect_bronze(full)
    print(f"[migração] {full}: colunas {list(missing)} acrescentadas, linhas existentes mantidas")
    return True


def ingest_bronze(df, name, source, comment=None):
    """Ingestão INCREMENTAL para a bronze (só append, nunca overwrite).

    Só acrescenta ficheiros que ainda não foram lidos (mesmo caminho e mesma data de
    modificação). Assim a bronze guarda o histórico de todas as leituras e pode
    voltar a correr sem duplicar dados. Tabelas de versões antigas são migradas.
    """
    full = tbl(BRONZE, name)
    df = with_audit(df, source)
    exists = spark.catalog.tableExists(full)
    if exists:
        migrate_bronze(full)
        history = spark.table(full).select("_source_file", "_file_modified").distinct()
        known = history.where(F.col("_file_modified").isNotNull())
        # linhas 'legado' não têm data do ficheiro: o mesmo caminho conta como já lido
        legacy_paths = history.where(F.col("_file_modified").isNull()).select("_source_file").distinct()
        df = (df.join(F.broadcast(known), ["_source_file", "_file_modified"], "left_anti")
                .join(F.broadcast(legacy_paths), "_source_file", "left_anti"))
    n_new = df.count()
    if n_new == 0:
        print(f"[bronze] {full}: sem ficheiros novos, nada acrescentado")
        return
    df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(full)
    if not exists:
        protect_bronze(full)
    set_comment(full, comment)
    print(f"[bronze] {full}: +{n_new:,} linhas (lote {BATCH_ID})")


def ingest_source_table(source, key, table, reader, comment):
    """Lê os ficheiros de uma fonte para a bronze, decidindo em função do que já existe:
    * há ficheiros no Volume            -> ingestão incremental;
    * não há ficheiros mas a tabela existe -> mantém o histórico já guardado (sem erro);
    * não há ficheiros nem tabela       -> erro com instruções.
    'reader' é uma função que recebe o caminho dos ficheiros e devolve um DataFrame
    (ex.: lambda p: spark.read.json(p)). Só é chamada se houver ficheiros."""
    full = tbl(BRONZE, table)
    if source_files(source, key):
        ingest_bronze(reader(source_glob(source, key)), table, source, comment)
    elif spark.catalog.tableExists(full):
        print(f"[bronze] {full}: não há '{FILES[source][key]}' no Volume; mantém-se o histórico já guardado")
    else:
        raise FileNotFoundError(f"Falta '{FILES[source][key]}' em {SOURCES[source]} e a tabela {full} ainda não existe. "
                                "Carrega o ficheiro em Catalog > bronze > raw > Upload to this volume.")


def latest(df, keys, order_cols=("_ingested_at",)):
    """Silver: fica só com a versão mais recente de cada chave (de forma determinística)."""
    w = Window.partitionBy(*keys).orderBy(*[F.col(c).desc() for c in order_cols])
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def nullify(df, tokens=("\\N",), cols=None):
    """Converte marcadores de 'em falta' (ex.: \\N do IMDb) em null, num único select."""
    target = set(cols or [c for c, t in df.dtypes if t == "string"])
    return df.select(*[
        F.when(F.col(c).isin(*tokens), F.lit(None)).otherwise(F.col(c)).alias(c) if c in target else F.col(c)
        for c in df.columns
    ])


def try_cast(col_name, dtype):
    """Cast tolerante: valores inválidos ficam null em vez de parar o job."""
    return F.expr(f"try_cast(`{col_name}` AS {dtype})")


def safe_date(col_name):
    """'yyyy-MM-dd' para data; qualquer outro valor fica null."""
    return F.when(F.col(col_name).rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(F.col(col_name)))


def imdb_tconst(col_name):
    """Constrói o tconst do IMDb (tt0114709). Preenche com zeros até 7 dígitos mas
    NUNCA corta: os filmes recentes já têm 8 dígitos (tt10872600)."""
    c = F.trim(F.col(col_name))
    return F.when(c.isNull() | (c == ""), F.lit(None)) \
            .otherwise(F.concat(F.lit("tt"), F.when(F.length(c) < 7, F.lpad(c, 7, "0")).otherwise(c)))


def write_delta(df, layer, name, cluster_by=None, comment=None):
    """Silver e gold: reescreve a tabela a partir da camada anterior (idempotente),
    opcionalmente com Liquid Clustering, e devolve o DataFrame lido da tabela."""
    full = tbl(layer, name)
    if cluster_by:
        view = f"_w_{uuid.uuid4().hex[:8]}"
        df.createOrReplaceTempView(view)
        spark.sql(f"CREATE OR REPLACE TABLE {full} CLUSTER BY ({', '.join(cluster_by)}) AS SELECT * FROM {view}")
        spark.catalog.dropTempView(view)
    else:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full)
    set_comment(full, comment)
    out = spark.table(full)
    print(f"[{layer}] {full}: {out.count():,} linhas")
    return out


def assert_unique(df, key, label):
    """Controlo de qualidade: a chave tem de ser única."""
    keys = [key] if isinstance(key, str) else list(key)
    total, distinct = df.agg(F.count("*"), F.countDistinct(*keys)).first()
    if total != distinct:
        raise ValueError(f"{label}: chave {keys} duplicada ({total:,} linhas, {distinct:,} distintas)")
    print(f"[qualidade] {label}: chave {keys} única ({total:,} linhas)")


def require_tables(layer, names, etapa_anterior):
    """Pré-requisito de uma etapa: as tabelas de que depende têm de existir."""
    missing = [n for n in names if not spark.catalog.tableExists(tbl(layer, n))]
    if missing:
        raise RuntimeError(f"Faltam as tabelas {layer}.{missing}. Corre primeiro a etapa '{etapa_anterior}' "
                           "(ou etapa = tudo).")


# ------------------------- 4. Pipeline visual ---------------------------------
# [Auxiliar] Desenho em SVG (texto que o browser transforma em imagem). Só serve para mostrar
# o estado do pipeline; não é preciso perceber o código de desenho.
def _layer_state(tables):
    """(nº tabelas existentes, nº esperadas, total de linhas)."""
    existing = [t for t in tables if spark.catalog.tableExists(t)]
    rows = sum(spark.table(t).count() for t in existing)
    return len(existing), len(tables), rows


def mostrar_pipeline(titulo="Estado do pipeline"):
    """Desenha o pipeline Medallion e pinta cada caixa conforme o estado das tabelas:
    verde = todas criadas, amarelo = parcial, cinzento = ainda por correr."""
    colors = {"full": ("#d4edda", "#2e7d32"), "partial": ("#fff3cd", "#b8860b"), "none": ("#eeeeee", "#9e9e9e")}

    def box(x, y, title, tables, w=180, h=64):
        done, total, rows = _layer_state(tables)
        state = "full" if done == total else ("partial" if done else "none")
        fill, stroke = colors[state]
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
                f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" font-weight="bold" font-size="13">{title}</text>'
                f'<text x="{x + w / 2}" y="{y + 44}" text-anchor="middle" font-size="11">{done}/{total} tabelas · {rows:,} linhas</text>')

    def file_box(x, y, title, source, w=150, h=64):
        try:
            ok = not missing_files(source)
        except Exception:
            ok = False
        fill, stroke = colors["full" if ok else "none"]
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
                f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" font-weight="bold" font-size="13">{title}</text>'
                f'<text x="{x + w / 2}" y="{y + 44}" text-anchor="middle" font-size="11">'
                f'{"ficheiros no Volume" if ok else "ficheiros em falta"}</text>')

    def arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="1.6" marker-end="url(#arr)"/>'

    rows_y = {"tmdb": 40, "imdb": 140, "movielens": 240}
    labels = {"tmdb": "TMDB (JSON)", "imdb": "IMDb (TSV)", "movielens": "MovieLens (CSV)"}
    parts = []
    for s, y in rows_y.items():
        parts.append(file_box(10, y, labels[s], s))
        parts.append(box(200, y, f"Bronze · {s}", [tbl(BRONZE, t) for t in BRONZE_TABLES[s]]))
        parts.append(box(420, y, f"Silver · {s}", [tbl(SILVER, t) for t in SILVER_TABLES[s]]))
        parts += [arrow(160, y + 32, 198, y + 32), arrow(380, y + 32, 418, y + 32), arrow(600, y + 32, 648, 172)]
    parts.append(box(650, 140, "Silver integrada", [tbl(SILVER, "movies_unified")]))
    parts.append(arrow(830, 172, 868, 172))
    parts.append(box(870, 140, "Gold", [tbl(GOLD, t) for t in GOLD_TABLES + STAR_TABLES]))
    for x, txt in [(85, "Fontes"), (290, "Bronze"), (510, "Silver"), (740, "Integração"), (960, "Gold")]:
        parts.append(f'<text x="{x}" y="25" text-anchor="middle" font-size="12" fill="#555">{txt}</text>')
    legend = ('<text x="10" y="335" font-size="11" fill="#333">'
              '<tspan fill="#2e7d32">■</tspan> criado · <tspan fill="#b8860b">■</tspan> parcial · '
              '<tspan fill="#9e9e9e">■</tspan> por correr</text>')
    displayHTML(f"""
    <div style="background:#fff;padding:8px;font-family:sans-serif;color:#222">
      <h3 style="margin:4px 0">{titulo} <span style="font-weight:normal;font-size:13px">(etapa: {ETAPA}, lote: {BATCH_ID})</span></h3>
      <svg width="1060" height="345" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">
        <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
        {''.join(parts)}{legend}
      </svg>
    </div>""")


# ------------------------- 5. Estilo dos gráficos -----------------------------
SOURCE_COLORS = {"IMDb": "#1f77b4", "MovieLens": "#ff7f0e", "TMDB": "#2ca02c"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def ci95(sd, n):
    """Meia largura do intervalo de confiança a 95% para uma média."""
    return 1.96 * sd / np.sqrt(n)


# ------------------------- 6. Resumo do ambiente ------------------------------
display(spark.createDataFrame(results, "verificacao string, estado string, detalhe string"))
_fails = [r for r in results if r[1] == "FAIL"]
if _fails:
    raise RuntimeError(f"Setup inválido: {len(_fails)} verificação(ões) crítica(s) falharam.")


# ------------------- 7. Objetos existentes e ação prevista --------------------
# [Auxiliar] Inventário: para cada tabela diz se já existe e o que o notebook vai fazer com ela.
def table_info(layer, name):
    """Estado atual de uma tabela: existe?, linhas, última alteração e colunas."""
    full = tbl(layer, name)
    if not spark.catalog.tableExists(full):
        return {"existe": False, "linhas": None, "ultima_alteracao": None, "colunas": []}
    df = spark.table(full)
    try:
        last = str(spark.sql(f"DESCRIBE DETAIL {full}").first()["lastModified"])
    except Exception:
        last = None
    return {"existe": True, "linhas": df.count(), "ultima_alteracao": last, "colunas": df.columns}


def source_of(name):
    return next((s for s, names in BRONZE_TABLES.items() if name in names), None)


def planned_action(layer, name, info):
    """Explica o que esta execução vai fazer à tabela, sem perder dados."""
    if layer == BRONZE:
        source = source_of(name)
        if not info["existe"]:
            return "será criada na primeira ingestão"
        if REINICIAR == "tudo_com_backup":
            if missing_files(source):
                return "MANTIDA: faltam ficheiros no Volume, por isso não é reiniciada"
            return "cópia de segurança em backup.* e depois recriada a partir dos ficheiros"
        missing = [c for c in BRONZE_AUDIT_COLS if c not in info["colunas"]]
        if missing:
            return f"versão antiga: será migrada (acrescenta {missing}, mantém as linhas)"
        return "mantida; só recebe ficheiros novos (append)"
    if layer == GOLD and name == "dq_results":
        if REINICIAR == "tudo_com_backup" and info["existe"]:
            return "cópia de segurança em backup.* e depois recriada"
        return "mantida; recebe novas verificações (append)"
    if not info["existe"]:
        return "será criada"
    if REINICIAR != "nao":
        return "apagada e recriada (é derivada da camada anterior)"
    return "reconstruída a partir da camada anterior (versão anterior fica no histórico Delta)"


EXPECTED_TABLES = ([(BRONZE, t) for names in BRONZE_TABLES.values() for t in names]
                   + [(SILVER, t) for names in SILVER_TABLES.values() for t in names] + [(SILVER, "movies_unified")]
                   + [(GOLD, t) for t in STAR_TABLES + GOLD_TABLES])
_objects = []
for _layer, _name in EXPECTED_TABLES:
    _info = table_info(_layer, _name)
    _objects.append((_layer, _name, _info["existe"], _info["linhas"], _info["ultima_alteracao"],
                     planned_action(_layer, _name, _info)))
print(f"Objetos existentes: {sum(o[2] for o in _objects)} de {len(_objects)} tabelas | reiniciar = {REINICIAR}")
display(spark.createDataFrame(_objects, "camada string, tabela string, existe boolean, linhas long, "
                                        "ultima_alteracao string, acao_prevista string"))


# ------------------- 8. Reinício seguro (opcional) ----------------------------
# [Auxiliar] Só é usado com o widget reiniciar; faz cópia de segurança antes de apagar.
def backup_table(full):
    """Copia a tabela para o schema backup antes de a apagar (DEEP CLONE; se não existir, CTAS)."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.backup")
    target = f"{CATALOG}.backup.{full.split('.')[-2]}_{full.split('.')[-1]}_{BATCH_ID}"
    try:
        spark.sql(f"CREATE TABLE {target} DEEP CLONE {full}")
    except Exception:
        spark.sql(f"CREATE TABLE {target} AS SELECT * FROM {full}")
    if spark.table(target).count() != spark.table(full).count():
        raise RuntimeError(f"Cópia de segurança de {full} incompleta: nada foi apagado.")
    return target


def drop_table(full, backup=False):
    if not spark.catalog.tableExists(full):
        return
    if backup:
        print(f"[backup] {full} -> {backup_table(full)}")
    spark.sql(f"DROP TABLE {full}")
    print(f"[reinício] {full} apagada")


if REINICIAR != "nao":
    with_backup = REINICIAR == "tudo_com_backup"
    # silver e gold são derivadas: podem ser apagadas sem perder informação
    for _layer, _name in EXPECTED_TABLES:
        if _layer == SILVER or (_layer == GOLD and _name != "dq_results"):
            drop_table(tbl(_layer, _name))
    if with_backup:
        drop_table(tbl(GOLD, "dq_results"), backup=True)
        for _source, _names in BRONZE_TABLES.items():
            if missing_files(_source):   # sem ficheiros não há como reconstruir: a tabela fica
                print(f"[reinício] bronze de {_source} mantida (faltam ficheiros no Volume)")
                continue
            for _name in _names:
                drop_table(tbl(BRONZE, _name), backup=True)
    print("Reinício concluído. Os ficheiros do Volume não foram alterados.")

print(f"Setup válido | etapa = {ETAPA} | lote = {BATCH_ID}")
mostrar_pipeline("Estado do pipeline antes de correr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Business Understanding (CRISP-DM fase 1)
# MAGIC
# MAGIC **Contexto.** Quem escolhe um filme olha para uma nota. Mas há várias: o IMDb (votos de qualquer pessoa registada), o TMDB (comunidade do The Movie Database) e o MovieLens (utilizadores de um sistema de recomendação académico). Uma plataforma de streaming que queira mostrar "a nota" de um filme precisa de saber se estas fontes concordam e quando não concordam.
# MAGIC
# MAGIC **Pergunta de negócio.** As notas dos filmes diferem entre o IMDb, o TMDB e os utilizadores do MovieLens? Que fatores (género, década, idioma, orçamento) explicam essas diferenças?
# MAGIC
# MAGIC **Hipóteses a testar**
# MAGIC * **H1:** as três fontes dão notas médias diferentes ao mesmo conjunto de filmes.
# MAGIC * **H2:** a diferença IMDb − MovieLens depende do género.
# MAGIC * **H3:** filmes em idioma não inglês têm diferenças maiores.
# MAGIC * **H4:** filmes de orçamento alto têm notas mais consensuais.
# MAGIC * **H5:** os filmes antigos têm notas mais altas (efeito de sobrevivência: só os clássicos continuam a ser vistos).
# MAGIC
# MAGIC **Critério de sucesso.** Responder a cada hipótese com uma tabela gold e um gráfico, e quantificar o efeito de cada fator num modelo de regressão.
# MAGIC
# MAGIC **Dados.** Estruturados: notas, votos, orçamento, receita, duração. Não estruturados: o JSON bruto do TMDB, as sinopses e as tags de texto livre do MovieLens.
# MAGIC
# MAGIC **Chave de integração.** O `links.csv` do MovieLens liga `movieId` ↔ `imdbId` ↔ `tmdbId`. O IMDb usa `tconst` (`tt0114709`), construído a partir do `imdbId`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Arquitetura da solução (Lakehouse + Medallion)
# MAGIC
# MAGIC O diagrama a cores é desenhado pela célula seguinte. Resumo por camada:
# MAGIC
# MAGIC | | Camada | Onde fica | O que contém | Regra principal |
# MAGIC |---|---|---|---|---|
# MAGIC | **Fontes**  | descarregadas localmente (a Free Edition não acede à internet) | IMDb (TSV), TMDB (JSON), MovieLens (CSV) | ficheiros originais, sem alterações |
# MAGIC | **Bronze**  | Volume `bronze.raw` + tabelas `bronze.*` | dados brutos em string + auditoria (`_batch_id`, `_source_file`, `_ingested_at`) | só **append**: guarda o histórico de todas as leituras |
# MAGIC | **Silver**  | tabelas `silver.*` | dados limpos, tipados, sem duplicados; `movies_unified` integra as 3 fontes | versão mais recente de cada chave; reconstruível a partir da bronze |
# MAGIC | **Gold**    | tabelas `gold.*` | agregados por década, género, idioma e orçamento; modelo; qualidade | uma tabela por hipótese, pronta para gráficos |
# MAGIC | **Consumo** | este notebook e o Job | 10 gráficos, regressão, conclusões | lê só a gold |

# COMMAND ----------

# DBTITLE 1,Cell 6
# [Auxiliar] Diagrama em HTML/CSS: cartões coloridos lado a lado. Não é preciso perceber o HTML.
def mostrar_arquitetura():
    """Diagrama da arquitetura (HTML/CSS: fica visível no notebook e no export HTML)."""
    layers = [
        ("Fontes", "#e8eef7", "#4c72b0", "", "Ficheiros descarregados localmente", [
            ("IMDb", "title.basics, title.ratings (TSV)"),
            ("TMDB", "listas, géneros, detalhe (JSON)"),
            ("MovieLens", "ratings, movies, links, tags (CSV)")]),
        ("Bronze", "#f6e7d8", "#a0522d", "", "Dados brutos com histórico de leituras", [
            ("Volume raw", "/Volumes/&lt;catálogo&gt;/bronze/raw/{imdb, tmdb, movielens}"),
            ("9 tabelas Delta", "tudo em string + _batch_id, _source_file, _ingested_at"),
            ("Regra", "só append de ficheiros novos")]),
        ("Silver", "#eef0f2", "#6c757d", "", "Dados limpos e integrados", [
            ("Por fonte", "imdb_movies, tmdb_movie_details, ml_ratings, ml_links, ml_tags…"),
            ("Integrada", "movies_unified (IMDb ⨝ MovieLens ⨝ TMDB)"),
            ("Qualidade", "chaves únicas, tipos, regras dq_results")]),
        ("Gold", "#fbf3d5", "#b8860b", "", "Pronta para responder às hipóteses", [
            ("Comparação", "ratings_by_decade, gap_by_genre"),
            ("Fatores", "gap_by_language, gap_by_budget, roi_by_genre"),
            ("Modelo e texto", "gap_model_coefficients, tags_by_gap")]),
        ("Consumo", "#e6f4ea", "#2e7d32", "", "Análise e decisão", [
            ("Gráficos", "10 figuras com intervalos de confiança"),
            ("Modelo", "regressão do gap IMDb − MovieLens"),
            ("Job", "pipeline agendável com tarefas por etapa")]),
    ]
    cards = []
    for i, (name, bg, border, icon, subtitle, items) in enumerate(layers):
        rows = "".join(f'<div style="margin:6px 0"><b>{k}</b><br><span style="color:#555;overflow-wrap:anywhere">{v}</span></div>'
                       for k, v in items)
        cards.append(
            f'<div style="flex:1 1 150px;min-width:150px;background:{bg};border-top:6px solid {border};'
            f'border-radius:10px;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.12)">'
            f'<div style="font-size:18px;font-weight:700;color:{border}">{icon} {name}</div>'
            f'<div style="font-size:12px;font-style:italic;color:#333;margin-bottom:6px">{subtitle}</div>'
            f'<div style="font-size:12px;color:#222">{rows}</div></div>')
        if i < len(layers) - 1:
            cards.append('<div style="align-self:center;font-size:22px;color:#888;padding:0">➜</div>')
    displayHTML(f"""
    <div style="background:#ffffff;color:#222;font-family:Segoe UI,Helvetica,Arial,sans-serif;padding:12px">
      <div style="font-size:20px;font-weight:700;margin-bottom:2px">Arquitetura Lakehouse · Medallion</div>
      <div style="font-size:13px;color:#555;margin-bottom:12px">
        Unity Catalog: <code>&lt;catálogo&gt;.bronze</code> · <code>.silver</code> · <code>.gold</code>
        &nbsp;|&nbsp; Delta Lake &nbsp;|&nbsp; Spark (serverless)</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:stretch">{''.join(cards)}</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#444;margin-top:12px">
        <span> <b>Incremental:</b> ficheiros já lidos não voltam a entrar</span>
        <span> <b>Sem perda de dados:</b> bronze só em append; histórico Delta e <code>RESTORE</code></span>
        <span> <b>Orquestração:</b> Job com tmdb ∥ imdb ∥ movielens → integração → gold → análise</span>
      </div>
    </div>""")


mostrar_arquitetura()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Understanding: ingestão para a Bronze (CRISP-DM fase 2)
# MAGIC
# MAGIC A bronze é **incremental**: cada execução acrescenta (`append`) apenas os ficheiros novos, marcados com `_batch_id`. Se o mesmo ficheiro for lido duas vezes, não é duplicado. Para simular uma segunda leitura da fonte, basta carregar um ficheiro novo para a pasta da fonte no Volume (ex.: `movie_details_2026-11-10.jsonl`) e voltar a correr: só esse ficheiro entra.
# MAGIC
# MAGIC Os ficheiros são lidos por pasta (`*.jsonl`, `*.tsv.gz`), por isso ficheiros novos com o mesmo formato são apanhados automaticamente.

# COMMAND ----------

# ----------------------------- BRONZE · TMDB ---------------------------------
def bronze_tmdb():
    # o 'payload' guarda o JSON completo como texto (componente não estruturada)
    ingest_source_table("tmdb", "lists", "tmdb_movie_lists",
                        lambda p: spark.read.schema("list_name string, page int, payload string").json(p),
                        "TMDB: listas popular/top_rated/now_playing (JSON bruto, histórico de leituras)")
    ingest_source_table("tmdb", "genres", "tmdb_genres",
                        lambda p: spark.read.schema("payload string").json(p),
                        "TMDB: lista de géneros (JSON bruto)")
    ingest_source_table("tmdb", "details", "tmdb_movie_details",
                        lambda p: spark.read.schema("movie_id long, payload string").json(p),
                        "TMDB: detalhe por filme (JSON bruto, histórico de leituras)")
if correr("tmdb"):
    bronze_tmdb()

# COMMAND ----------

# ----------------------------- BRONZE · IMDb ---------------------------------
BASICS_COLS  = ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                "startYear", "endYear", "runtimeMinutes", "genres"]
RATINGS_COLS = ["tconst", "averageRating", "numVotes"]


def bronze_imdb():
    # TSV do IMDb: separador tab e sem aspas (os títulos podem conter aspas)
    ingest_source_table("imdb", "basics", "imdb_title_basics",
                        lambda p: read_raw_csv(p, BASICS_COLS, sep="\t", quote=""),
                        "IMDb title.basics (só filmes), valores brutos em string")
    ingest_source_table("imdb", "ratings", "imdb_title_ratings",
                        lambda p: read_raw_csv(p, RATINGS_COLS, sep="\t", quote=""),
                        "IMDb title.ratings (só filmes), valores brutos em string")


if correr("imdb"):
    bronze_imdb()

# COMMAND ----------

# --------------------------- BRONZE · MovieLens -------------------------------
ML_SCHEMAS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies":  ["movieId", "title", "genres"],
    "links":   ["movieId", "imdbId", "tmdbId"],
    "tags":    ["userId", "movieId", "tag", "timestamp"],
}


def bronze_movielens():
    # CSV com aspas duplicadas dentro dos títulos e tags, daí escape='"'.
    # movies e tags podem ter texto com quebras de linha, daí multiline=True.
    ingest_source_table("movielens", "ratings", "ml_ratings",
                        lambda p: read_raw_csv(p, ML_SCHEMAS["ratings"], escape='"'),
                        "MovieLens ratings.csv, valores brutos em string")
    ingest_source_table("movielens", "movies", "ml_movies",
                        lambda p: read_raw_csv(p, ML_SCHEMAS["movies"], escape='"', multiline=True),
                        "MovieLens movies.csv, valores brutos em string")
    ingest_source_table("movielens", "links", "ml_links",
                        lambda p: read_raw_csv(p, ML_SCHEMAS["links"], escape='"'),
                        "MovieLens links.csv, valores brutos em string")
    ingest_source_table("movielens", "tags", "ml_tags",
                        lambda p: read_raw_csv(p, ML_SCHEMAS["tags"], escape='"', multiline=True),
                        "MovieLens tags.csv, valores brutos em string")


if correr("movielens"):
    bronze_movielens()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Perfil dos dados brutos
# MAGIC
# MAGIC O gráfico mostra quantas linhas entraram em cada tabela bronze em cada lote. Com duas leituras da fonte aparecem duas cores na mesma barra: é a evidência de que a bronze guarda o histórico de leituras.

# COMMAND ----------

if correr("analise"):
    _parts = []
    for _src, _names in BRONZE_TABLES.items():
        for _name in _names:
            if spark.catalog.tableExists(tbl(BRONZE, _name)):
                _t = spark.table(tbl(BRONZE, _name))
                _lote = F.col("_batch_id") if "_batch_id" in _t.columns else F.lit("legado")
                _parts.append(_t.groupBy(_lote.alias("_batch_id")).count().withColumn("tabela", F.lit(_name)))
    if _parts:
        _df = _parts[0]
        for _p in _parts[1:]:
            _df = _df.unionByName(_p)
        batches = _df.toPandas().pivot_table(index="tabela", columns="_batch_id", values="count", fill_value=0)
 
        fig, ax = plt.subplots(figsize=(10, 4.5))
        batches.plot.barh(ax=ax, colormap="tab10", width=0.8)   # uma barra por lote
        ax.set_xscale("log")
        ax.set_xlim(left=0.5)
        # Eixo em milhões: 1 000 000 -> 1M, 100 000 -> 0,1M (abaixo de 1000 mostra o número normal)
        def em_milhoes(x, _):
            if x < 1000:
                return f"{x:.0f}"
            return f"{x / 1e6:g}M".replace(".", ",")
        ax.xaxis.set_major_formatter(FuncFormatter(em_milhoes))
        ax.set_xlabel("Linhas, em milhões (M), escala logarítmica")
        ax.set_ylabel("")
        ax.set_title("Figura 1. Linhas por tabela bronze e por lote de ingestão")
        ax.legend(title="Lote", fontsize=8, loc="lower right")
        fig.tight_layout()
        plt.show()
        print(f"Lotes distintos na bronze: {batches.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Preparation: Silver (CRISP-DM fase 3)
# MAGIC
# MAGIC A silver é reconstruída a partir de **todo** o histórico da bronze, ficando com a versão mais recente de cada chave (`latest`). Por isso é sempre correta, mesmo depois de várias leituras da fonte.
# MAGIC
# MAGIC Regras aplicadas: `\N` e strings vazias passam a `null`; tipagem com `try_cast`; deduplicação determinística; géneros em array; datas validadas.

# COMMAND ----------

# ----------------------------- SILVER · TMDB ---------------------------------
LIST_MOVIE_DDL = ("id long, title string, original_title string, original_language string, overview string, "
                  "release_date string, popularity double, vote_average double, vote_count long")
DETAIL_DDL = ("id long, imdb_id string, title string, original_title string, original_language string, "
              "overview string, release_date string, runtime int, budget long, revenue long, status string, "
              "popularity double, vote_average double, vote_count long, "
              "genres array<struct<id:int,name:string>>")


def silver_tmdb():
    require_tables(BRONZE, ["tmdb_movie_lists", "tmdb_movie_details"], "tmdb")
    # Filmes das listas: a versão mais recente de cada filme + todas as listas onde apareceu
    lists = (spark.table(tbl(BRONZE, "tmdb_movie_lists"))
             .select("list_name", "_ingested_at", F.from_json("payload", LIST_MOVIE_DDL).alias("m"))
             .select("list_name", "_ingested_at", F.col("m.id").alias("tmdb_id"), "m.*")
             .where(F.col("tmdb_id").isNotNull()))
    lists_per_movie = lists.groupBy("tmdb_id").agg(F.sort_array(F.collect_set("list_name")).alias("lists"))
    tmdb_movies = (latest(lists, ["tmdb_id"])
                   .join(lists_per_movie, "tmdb_id")
                   .select("tmdb_id", "title", "original_title", "original_language", "overview",
                           safe_date("release_date").alias("release_date"),
                           "popularity", "vote_average", "vote_count", "lists"))
    assert_unique(tmdb_movies, "tmdb_id", "silver.tmdb_movies")
    write_delta(tmdb_movies, SILVER, "tmdb_movies",
                comment="TMDB: filmes das listas, um por tmdb_id, com as listas em que aparecem")

    # Detalhe por filme: a versão mais recente de cada tmdb_id
    details = (spark.table(tbl(BRONZE, "tmdb_movie_details"))
               .select("_ingested_at", F.from_json("payload", DETAIL_DDL).alias("d"))
               .select("_ingested_at", "d.*")
               .where(F.col("id").isNotNull()))
    tmdb_details = (latest(nullify(details, tokens=("",)), ["id"])
                    .select(F.col("id").alias("tmdb_id"), "imdb_id", "title", "original_title",
                            "original_language", "overview", safe_date("release_date").alias("release_date"),
                            F.col("runtime").alias("runtime_minutes"), "budget", "revenue", "status",
                            "popularity", "vote_average", "vote_count", "genres"))
    assert_unique(tmdb_details, "tmdb_id", "silver.tmdb_movie_details")
    tmdb_details = write_delta(tmdb_details, SILVER, "tmdb_movie_details",
                               comment="TMDB: detalhe tipado (orçamento, receita, duração, géneros), versão mais recente")

    tmdb_genres = (tmdb_details.select("tmdb_id", F.explode("genres").alias("g"))
                   .select("tmdb_id", F.col("g.id").alias("genre_id"), F.col("g.name").alias("genre_name")))
    write_delta(tmdb_genres, SILVER, "tmdb_movie_genres", comment="TMDB: relação filme-género")


if correr("tmdb"):
    silver_tmdb()

# COMMAND ----------

# ----------------------------- SILVER · IMDb ---------------------------------
def silver_imdb():
    require_tables(BRONZE, ["imdb_title_basics", "imdb_title_ratings"], "imdb")
    basics  = latest(nullify(spark.table(tbl(BRONZE, "imdb_title_basics"))), ["tconst"])
    ratings = latest(nullify(spark.table(tbl(BRONZE, "imdb_title_ratings"))), ["tconst"])

    imdb_ratings = ratings.select("tconst",
                                  try_cast("averageRating", "double").alias("imdb_rating"),
                                  try_cast("numVotes", "bigint").alias("imdb_votes"))
    imdb_movies = (
        basics.where(F.col("titleType") == "movie")
        .select("tconst",
                F.col("primaryTitle").alias("title"),
                F.col("originalTitle").alias("original_title"),
                F.coalesce(F.col("isAdult") == "1", F.lit(False)).alias("is_adult"),
                try_cast("startYear", "int").alias("start_year"),
                try_cast("runtimeMinutes", "int").alias("runtime_minutes"),
                F.split("genres", ",").alias("genres"))
        .where(~F.col("is_adult"))
        .join(imdb_ratings, "tconst", "left")
    )
    assert_unique(imdb_movies, "tconst", "silver.imdb_movies")
    imdb_movies = write_delta(imdb_movies, SILVER, "imdb_movies", cluster_by=["start_year"],
                              comment="IMDb: filmes não adultos, tipados, com nota e votos (Liquid Clustering por start_year)")
    write_delta(imdb_movies.select("tconst", F.explode("genres").alias("genre")), SILVER, "imdb_movie_genres",
                comment="IMDb: relação filme-género (nomes em inglês)")


if correr("imdb"):
    silver_imdb()

# COMMAND ----------

# --------------------------- SILVER · MovieLens -------------------------------
YEAR_PATTERN = r"\((\d{4})\)\s*$"


def silver_movielens():
    require_tables(BRONZE, ["ml_ratings", "ml_movies", "ml_links", "ml_tags"], "movielens")
    # Avaliações: uma por (utilizador, filme); se houver repetidas fica a mais recente
    ml_ratings = (
        spark.table(tbl(BRONZE, "ml_ratings"))
        .select(try_cast("userId", "int").alias("user_id"),
                try_cast("movieId", "int").alias("movie_id"),
                try_cast("rating", "double").alias("rating"),
                F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("rated_at"),
                "_ingested_at")
        .where(F.col("user_id").isNotNull() & F.col("movie_id").isNotNull()
               & F.col("rating").between(0.5, 5.0))
    )
    ml_ratings = latest(ml_ratings, ["user_id", "movie_id"], ("_ingested_at", "rated_at")).drop("_ingested_at")
    assert_unique(ml_ratings, ["user_id", "movie_id"], "silver.ml_ratings")
    write_delta(ml_ratings, SILVER, "ml_ratings", cluster_by=["movie_id"],
                comment="MovieLens: uma avaliação por utilizador e filme (escala 0.5 a 5)")

    ml_movies = (
        latest(spark.table(tbl(BRONZE, "ml_movies")), ["movieId"])
        .select(try_cast("movieId", "int").alias("movie_id"),
                F.trim(F.regexp_replace("title", YEAR_PATTERN, "")).alias("title"),
                F.when(F.regexp_extract("title", YEAR_PATTERN, 1) != "",
                       F.regexp_extract("title", YEAR_PATTERN, 1).cast("int")).alias("year"),
                F.filter(F.split("genres", r"\|"),
                         lambda g: (g != "") & (g != "(no genres listed)")).alias("genres"))
        .where(F.col("movie_id").isNotNull())
    )
    assert_unique(ml_movies, "movie_id", "silver.ml_movies")
    write_delta(ml_movies, SILVER, "ml_movies", comment="MovieLens: filmes com título, ano e géneros em array")

    ml_links = (
        latest(spark.table(tbl(BRONZE, "ml_links")), ["movieId"])
        .select(try_cast("movieId", "int").alias("movie_id"),
                imdb_tconst("imdbId").alias("tconst"),          # chave do IMDb (sem cortar 8 dígitos)
                try_cast("tmdbId", "int").alias("tmdb_id"))     # chave do TMDB
        .where(F.col("movie_id").isNotNull())
    )
    assert_unique(ml_links, "movie_id", "silver.ml_links")
    write_delta(ml_links, SILVER, "ml_links", comment="MovieLens: ponte movie_id <-> tconst (IMDb) <-> tmdb_id (TMDB)")

    ml_tags = (
        spark.table(tbl(BRONZE, "ml_tags"))
        .select(try_cast("userId", "int").alias("user_id"),
                try_cast("movieId", "int").alias("movie_id"),
                F.lower(F.trim("tag")).alias("tag"),
                F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("tagged_at"))
        .where(F.col("tag").isNotNull() & (F.col("tag") != ""))
        .dropDuplicates(["user_id", "movie_id", "tag"])
    )
    write_delta(ml_tags, SILVER, "ml_tags", comment="MovieLens: tags de texto livre normalizadas (minúsculas)")


if correr("movielens"):
    silver_movielens()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Silver integrada: `silver.movies_unified`
# MAGIC
# MAGIC Junta as três fontes por chave. O IMDb é a tabela grande e as restantes são pequenas, por isso usam-se **Broadcast Hash Joins** (o `explain` abaixo mostra-o).
# MAGIC Valores `0` de orçamento, receita e votos do TMDB significam "desconhecido" e passam a `null`. A nota do MovieLens (0.5 a 5) é multiplicada por 2 para ficar na escala do IMDb (1 a 10).

# COMMAND ----------

def silver_integracao():
    require_tables(SILVER, ["imdb_movies", "ml_links", "ml_ratings", "tmdb_movie_details"], "tmdb, imdb e movielens")
    imdb  = spark.table(tbl(SILVER, "imdb_movies"))
    links = spark.table(tbl(SILVER, "ml_links")).where(F.col("tconst").isNotNull())
    tmdb  = spark.table(tbl(SILVER, "tmdb_movie_details"))

    ml_stats = (spark.table(tbl(SILVER, "ml_ratings"))
                .groupBy("movie_id")
                .agg(F.avg("rating").alias("ml_avg_rating"), F.count("*").alias("ml_num_ratings")))

    tmdb_sel = tmdb.select(
        "tmdb_id", "original_language", "overview",
        F.when(F.col("vote_count") > 0, F.col("vote_average")).alias("tmdb_vote_average"),
        F.when(F.col("vote_count") > 0, F.col("vote_count")).alias("tmdb_vote_count"),
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
                    (F.floor(F.col("start_year") / 10) * 10).cast("int").alias("decade"),
                    "runtime_minutes", "genres", "imdb_rating", "imdb_votes",
                    F.round("ml_avg_rating", 3).alias("ml_avg_rating"),
                    F.round(F.col("ml_avg_rating") * 2, 3).alias("ml_rating_10"), "ml_num_ratings",
                    "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity",
                    "budget", "revenue", "original_language", "overview",
                    F.coalesce(F.col("in_tmdb"), F.lit(False)).alias("in_tmdb"))
    )
    unified.explain(mode="formatted")  # evidência do BroadcastHashJoin para o relatório

    assert_unique(unified, "movie_id", "silver.movies_unified")
    write_delta(unified, SILVER, "movies_unified", cluster_by=["decade", "start_year"],
                comment="Filmes integrados (IMDb + MovieLens + TMDB); ml_rating_10 = nota MovieLens na escala 1-10")


if correr("integracao"):
    silver_integracao()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Qualidade dos dados
# MAGIC
# MAGIC Cada regra conta as linhas que a violam. O resultado fica guardado em `gold.dq_results` (em `append`, com o lote), para se ver a evolução da qualidade ao longo das execuções.

# COMMAND ----------

DQ_RULES = [
    ("silver.imdb_movies",        "nota IMDb fora de 1-10",        "imdb_rating IS NOT NULL AND imdb_rating NOT BETWEEN 1 AND 10"),
    ("silver.imdb_movies",        "ano de estreia em falta",       "start_year IS NULL"),
    ("silver.ml_ratings",         "nota MovieLens fora de 0.5-5",  "rating NOT BETWEEN 0.5 AND 5"),
    ("silver.ml_links",           "filme sem tconst",              "tconst IS NULL"),
    ("silver.ml_links",           "filme sem tmdb_id",             "tmdb_id IS NULL"),
    ("silver.tmdb_movie_details", "orçamento ou receita negativos", "budget < 0 OR revenue < 0"),
    ("silver.tmdb_movie_details", "orçamento suspeito (< 1000 USD)", "budget > 0 AND budget < 1000"),
    ("silver.movies_unified",     "filme sem avaliações MovieLens", "ml_num_ratings IS NULL"),
    ("silver.movies_unified",     "filme sem detalhe TMDB",         "NOT in_tmdb"),
]


def run_quality_checks():
    require_tables(SILVER, ["movies_unified"], "integracao")
    rows = []
    for table, rule, condition in DQ_RULES:
        df = spark.table(f"{CATALOG}.{table}")
        total, bad = df.agg(F.count("*"), F.sum(F.expr(condition).cast("int"))).first()
        rows.append((table, rule, int(total), int(bad or 0)))
    dq = (spark.createDataFrame(rows, "tabela string, regra string, linhas long, violacoes long")
          .withColumn("pct_violacoes", F.round(100 * F.col("violacoes") / F.col("linhas"), 2))
          .withColumn("_batch_id", F.lit(BATCH_ID))
          .withColumn("_checked_at", F.current_timestamp()))
    dq.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(tbl(GOLD, "dq_results"))
    return dq


if correr("integracao"):
    display(run_quality_checks())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Modeling: tabelas Gold (CRISP-DM fase 4)
# MAGIC
# MAGIC Duas populações de análise, ambas com limiares mínimos de votos para evitar ruído (widgets `min_*`):
# MAGIC
# MAGIC * **`base`**: filmes com nota IMDb e MovieLens fiáveis. Usada para comparar IMDb com MovieLens.
# MAGIC * **`base_cmp`**: subconjunto da `base` que também tem nota TMDB fiável. Usada sempre que se comparam **as três fontes**, para que cada média seja calculada sobre **os mesmos filmes**.
# MAGIC
# MAGIC Diferença (gap) = nota IMDb − nota MovieLens (escala 1-10). Gap positivo significa que o IMDb é mais generoso.

# COMMAND ----------

def gold_bases():
    unified = spark.table(tbl(SILVER, "movies_unified"))
    base = (unified
            .where((F.col("imdb_votes") >= MIN_IMDB_VOTES)
                   & (F.col("ml_num_ratings") >= MIN_ML_RATINGS)
                   & F.col("decade").isNotNull())
            .withColumn("gap_imdb_ml", F.col("imdb_rating") - F.col("ml_rating_10")))
    base_cmp = (base.where(F.col("tmdb_vote_count") >= MIN_TMDB_VOTES)
                .withColumn("gap_imdb_tmdb", F.col("imdb_rating") - F.col("tmdb_vote_average")))
    return unified, base, base_cmp


def stats(col, prefix):
    """Média, desvio-padrão e nº de valores de uma coluna."""
    return [F.round(F.avg(col), 3).alias(f"avg_{prefix}"),
            F.round(F.stddev(col), 3).alias(f"sd_{prefix}")]


def gold_tables():
    require_tables(SILVER, ["movies_unified", "ml_tags"], "integracao")
    unified, base, base_cmp = gold_bases()
    print(f"base: {base.count():,} filmes | base_cmp (com TMDB): {base_cmp.count():,} filmes")

    # H1 e H5: nota média por fonte e por década, nos mesmos filmes
    by_decade = (base_cmp.groupBy("decade")
                 .agg(F.count("*").alias("n_movies"),
                      *stats("imdb_rating", "imdb"), *stats("ml_rating_10", "ml"), *stats("tmdb_vote_average", "tmdb"))
                 .where("n_movies >= 5"))
    write_delta(by_decade, GOLD, "ratings_by_decade",
                comment="Nota média e desvio-padrão por fonte e década, nos mesmos filmes (base_cmp)")

    # H2: gap IMDb - MovieLens por género
    by_genre = (base.withColumn("genre", F.explode("genres"))
                .groupBy("genre")
                .agg(F.count("*").alias("n_movies"), *stats("gap_imdb_ml", "gap"),
                     F.round(F.avg("imdb_rating"), 3).alias("avg_imdb"),
                     F.round(F.avg("ml_rating_10"), 3).alias("avg_ml"))
                .where("n_movies >= 20"))
    write_delta(by_genre, GOLD, "gap_by_genre", comment="Diferença IMDb - MovieLens por género (base, mín. 20 filmes)")

    # H3: gap por idioma original (do TMDB)
    by_language = (base_cmp.groupBy("original_language")
                   .agg(F.count("*").alias("n_movies"), *stats("gap_imdb_ml", "gap_ml"), *stats("gap_imdb_tmdb", "gap_tmdb"))
                   .where("n_movies >= 10"))
    write_delta(by_language, GOLD, "gap_by_language", comment="Diferenças de nota por idioma original (base_cmp, mín. 10 filmes)")

    # H4: gap e consenso por escalão de orçamento
    by_budget = (base.where(F.col("budget") >= 100_000)
                 .withColumn("budget_band",
                             F.when(F.col("budget") < 5e6, "1) < 5M")
                              .when(F.col("budget") < 30e6, "2) 5-30M")
                              .when(F.col("budget") < 100e6, "3) 30-100M")
                              .otherwise("4) >= 100M"))
                 .groupBy("budget_band")
                 .agg(F.count("*").alias("n_movies"), *stats("gap_imdb_ml", "gap"),
                      F.round(F.avg(F.abs("gap_imdb_ml")), 3).alias("avg_abs_gap")))
    write_delta(by_budget, GOLD, "gap_by_budget", comment="Diferença IMDb - MovieLens por escalão de orçamento (USD nominais)")

    # Filmes onde o IMDb e o MovieLens mais divergem
    gaps = base.select("movie_id", "tconst", "title", "start_year", "decade", "genres",
                       "imdb_rating", "imdb_votes", "ml_rating_10", "ml_num_ratings",
                       F.round("gap_imdb_ml", 2).alias("gap_imdb_ml"),
                       F.round(F.abs("gap_imdb_ml"), 2).alias("abs_gap"))
    write_delta(gaps, GOLD, "movie_rating_gaps", cluster_by=["start_year"],
                comment="Diferença entre a nota IMDb e a nota MovieLens (escala 1-10) por filme")

    # Retorno financeiro por género (orçamento mínimo de 100 mil USD para excluir valores lixo)
    roi = (unified.where((F.col("budget") >= 100_000) & F.col("revenue").isNotNull())
           .withColumn("genre", F.explode("genres"))
           .withColumn("roi", F.col("revenue") / F.col("budget"))
           .groupBy("genre")
           .agg(F.count("*").alias("n_movies"),
                F.round(F.percentile_approx("roi", 0.5), 2).alias("median_roi"),
                F.round(F.percentile_approx("roi", 0.25), 2).alias("p25_roi"),
                F.round(F.percentile_approx("roi", 0.75), 2).alias("p75_roi"),
                F.round(100 * F.avg((F.col("roi") > 1).cast("int")), 1).alias("pct_receita_maior_orcamento"),
                F.round(F.avg("imdb_rating"), 2).alias("avg_imdb_rating"))
           .where("n_movies >= 10"))
    write_delta(roi, GOLD, "roi_by_genre", comment="Retorno (receita/orçamento) por género: mediana e quartis, orçamento >= 100k USD")

    # Dados não estruturados: tags dos filmes onde as fontes discordam
    groups = (spark.table(tbl(GOLD, "movie_rating_gaps"))
              .withColumn("grupo", F.when(F.col("gap_imdb_ml") >= 1, "IMDb mais generoso")
                                    .when(F.col("gap_imdb_ml") <= -1, "MovieLens mais generoso"))
              .where(F.col("grupo").isNotNull())
              .select("movie_id", "grupo"))
    tags = (spark.table(tbl(SILVER, "ml_tags")).join(F.broadcast(groups), "movie_id")
            .groupBy("grupo", "tag").agg(F.countDistinct("movie_id").alias("n_movies"))
            .where("n_movies >= 3"))
    write_delta(tags, GOLD, "tags_by_gap", comment="Tags MovieLens mais frequentes nos filmes onde IMDb e MovieLens divergem >= 1 ponto")


if correr("gold"):
    gold_tables()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Modelo explicativo: que fatores explicam a diferença de notas?
# MAGIC
# MAGIC Regressão linear (mínimos quadrados) do **gap IMDb − MovieLens** sobre década, idioma, popularidade (log dos votos IMDb) e os 8 géneros mais frequentes. Cada coeficiente diz quantos pontos de gap esse fator acrescenta **mantendo os outros constantes**, o que as tabelas por grupo não conseguem.
# MAGIC
# MAGIC As features são preparadas em Spark. O ajuste é feito em pandas/numpy porque a tabela tem poucos milhares de linhas (já filtrada e agregada). Com dados maiores usar-se-ia `pyspark.ml.regression.LinearRegression`.

# COMMAND ----------

def ols(pdf, target, features):
    """Regressão linear com erros-padrão e p-values (aproximação normal, adequada para n grande)."""
    X = np.column_stack([np.ones(len(pdf)), pdf[features].to_numpy(dtype=float)])
    y = pdf[target].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    t = beta / se
    p = [math.erfc(abs(v) / math.sqrt(2)) for v in t]
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    out = pd.DataFrame({"variavel": ["(constante)"] + features, "coeficiente": beta,
                        "erro_padrao": se, "t": t, "p_value": p})
    return out, r2, n


def gold_model():
    require_tables(GOLD, ["gap_by_genre"], "gold")
    _, _, base_cmp = gold_bases()
    top_genres = [r.genre for r in spark.table(tbl(GOLD, "gap_by_genre"))
                  .orderBy(F.desc("n_movies")).limit(8).collect()]
    genre_cols = [f"genero_{g.replace('-', '_')}" for g in top_genres]
    feats = (base_cmp.select(
                "gap_imdb_ml",
                ((F.col("decade") - 2000) / 10).alias("decadas_desde_2000"),
                (F.col("original_language") != "en").cast("int").alias("idioma_nao_ingles"),
                F.log10("imdb_votes").alias("log10_votos_imdb"),
                *[F.coalesce(F.array_contains("genres", g), F.lit(False)).cast("int").alias(c)
                  for g, c in zip(top_genres, genre_cols)])
             .dropna())
    features = ["decadas_desde_2000", "idioma_nao_ingles", "log10_votos_imdb"] + genre_cols
    pdf = feats.toPandas()  # poucos milhares de linhas: seguro
    if len(pdf) < 100:
        print(f"Aviso: só {len(pdf)} filmes com as três fontes; o modelo é pouco fiável. "
              "Baixar min_tmdb_votes ou descarregar mais detalhe do TMDB.")
    coefs, r2, n = ols(pdf, "gap_imdb_ml", features)
    coefs["significativo_5pct"] = coefs["p_value"] < 0.05
    write_delta(spark.createDataFrame(coefs).withColumn("r2", F.lit(float(r2))).withColumn("n", F.lit(int(n))),
                GOLD, "gap_model_coefficients",
                comment="Regressão linear do gap IMDb - MovieLens: coeficientes, erros-padrão e p-values")
    print(f"n = {n:,} filmes | R² = {r2:.3f}")


if correr("gold"):
    gold_model()
    display(spark.table(tbl(GOLD, "gap_model_coefficients")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Modelo dimensional: star schema na Gold
# MAGIC
# MAGIC As tabelas gold anteriores são **agregados** prontos para cada hipótese. O **star schema** (modelo de Kimball) guarda os mesmos dados ao **detalhe**, organizados para qualquer pergunta futura e para ferramentas de BI (Dashboards do Databricks, Power BI):
# MAGIC
# MAGIC * **Tabela de factos `fact_rating`**: uma linha por **filme × fonte** (grão). Medidas: nota na escala 1-10 e número de votos. Pôr a fonte como dimensão é o que permite comparar IMDb, MovieLens e TMDB com um simples `GROUP BY`.
# MAGIC * **Dimensões**: `dim_movie` (título, duração, orçamento, receita, escalão de orçamento), `dim_source` (fonte e escala original), `dim_year` (ano e década), `dim_language` (idioma original).
# MAGIC * **Ponte `bridge_movie_genre`**: um filme tem vários géneros (relação muitos-para-muitos), por isso os géneros ligam-se através de uma tabela ponte para `dim_genre`.
# MAGIC
# MAGIC As chaves primárias e estrangeiras são registadas no Unity Catalog (**informativas**: documentam o modelo e aparecem no Catalog Explorer, mas não impedem duplicados). A unicidade é garantida pelo `assert_unique` antes da escrita.

# COMMAND ----------

SOURCE_DIM = [(1, "IMDb", "1 a 10", "nota original"),
              (2, "MovieLens", "0,5 a 5", "nota × 2"),
              (3, "TMDB", "0 a 10", "nota original")]

# (tabela, coluna) -> (tabela referenciada, coluna)
STAR_FKS = {("fact_rating", "movie_key"): ("dim_movie", "movie_key"),
            ("fact_rating", "source_key"): ("dim_source", "source_key"),
            ("fact_rating", "year_key"): ("dim_year", "year_key"),
            ("fact_rating", "language_key"): ("dim_language", "language_key"),
            ("bridge_movie_genre", "movie_key"): ("dim_movie", "movie_key"),
            ("bridge_movie_genre", "genre_key"): ("dim_genre", "genre_key")}
STAR_PKS = {"dim_movie": "movie_key", "dim_source": "source_key", "dim_year": "year_key",
            "dim_language": "language_key", "dim_genre": "genre_key"}


# [Auxiliar] Chaves primárias (PK) e estrangeiras (FK) do star schema. No Databricks são
# informativas: documentam as relações e ajudam o otimizador, mas não bloqueiam dados errados.
def _try_sql(statement):
    """Constraints são informativas: se o ambiente não as suportar, o pipeline continua."""
    try:
        spark.sql(statement)
        return True
    except Exception as e:
        print(f"[aviso] {statement[:70]}... -> {type(e).__name__}")
        return False


def drop_star_constraints():
    """Antes de reescrever as tabelas, remove as constraints (primeiro as FKs, depois as PKs)."""
    for (table, col), _ in STAR_FKS.items():
        if spark.catalog.tableExists(tbl(GOLD, table)):
            _try_sql(f"ALTER TABLE {tbl(GOLD, table)} DROP CONSTRAINT IF EXISTS fk_{table}_{col}")
    for table in STAR_PKS:
        if spark.catalog.tableExists(tbl(GOLD, table)):
            _try_sql(f"ALTER TABLE {tbl(GOLD, table)} DROP PRIMARY KEY IF EXISTS CASCADE")


def add_star_constraints():
    for table, col in STAR_PKS.items():
        if _try_sql(f"ALTER TABLE {tbl(GOLD, table)} ALTER COLUMN {col} SET NOT NULL"):
            _try_sql(f"ALTER TABLE {tbl(GOLD, table)} ADD CONSTRAINT pk_{table} PRIMARY KEY ({col})")
    for (table, col), (ref, ref_col) in STAR_FKS.items():
        _try_sql(f"ALTER TABLE {tbl(GOLD, table)} ADD CONSTRAINT fk_{table}_{col} "
                 f"FOREIGN KEY ({col}) REFERENCES {tbl(GOLD, ref)} ({ref_col})")


def gold_star():
    require_tables(SILVER, ["movies_unified"], "integracao")
    unified = spark.table(tbl(SILVER, "movies_unified"))
    drop_star_constraints()

    # ---- Dimensões ----
    dim_movie = unified.select(
        F.col("movie_id").alias("movie_key"), "tconst", "tmdb_id", "title", "runtime_minutes",
        "budget", "revenue",
        F.when(F.col("budget").isNull(), "desconhecido")
         .when(F.col("budget") < 5e6, "1) < 5M").when(F.col("budget") < 30e6, "2) 5-30M")
         .when(F.col("budget") < 100e6, "3) 30-100M").otherwise("4) >= 100M").alias("budget_band"))
    dim_source = spark.createDataFrame(SOURCE_DIM, "source_key int, source_name string, "
                                                   "escala_original string, conversao_1_10 string")
    dim_year = (unified.where(F.col("start_year").isNotNull())
                .select(F.col("start_year").alias("year_key"), "decade").distinct())
    lang_map = F.create_map(*[F.lit(x) for kv in LANG_NAMES.items() for x in kv])
    dim_language = (unified.select(F.coalesce("original_language", F.lit("n/d")).alias("language_key")).distinct()
                    .withColumn("language_name", F.coalesce(lang_map[F.col("language_key")], F.col("language_key"))))
    movie_genres = unified.select(F.col("movie_id").alias("movie_key"), F.explode("genres").alias("genre_name"))
    dim_genre = (movie_genres.select("genre_name").distinct()
                 .withColumn("genre_key", F.row_number().over(Window.orderBy("genre_name")))
                 .select("genre_key", "genre_name"))
    bridge = (movie_genres.join(F.broadcast(dim_genre), "genre_name")
              .select("movie_key", "genre_key").distinct())

    # ---- Factos: um filme dá até 3 linhas (uma por fonte com nota) ----
    keys = [F.col("movie_id").alias("movie_key"), F.col("start_year").alias("year_key"),
            F.coalesce("original_language", F.lit("n/d")).alias("language_key")]
    fact = (unified.select(*keys, F.lit(1).alias("source_key"),
                           F.col("imdb_rating").alias("rating_10"), F.col("imdb_votes").alias("n_votes"))
            .unionByName(unified.select(*keys, F.lit(2).alias("source_key"),
                                        F.col("ml_rating_10").alias("rating_10"), F.col("ml_num_ratings").alias("n_votes")))
            .unionByName(unified.select(*keys, F.lit(3).alias("source_key"),
                                        F.col("tmdb_vote_average").alias("rating_10"), F.col("tmdb_vote_count").alias("n_votes")))
            .where(F.col("rating_10").isNotNull()))

    # ---- Controlo de qualidade e escrita ----
    for df, key, name in [(dim_movie, "movie_key", "dim_movie"), (dim_source, "source_key", "dim_source"),
                          (dim_year, "year_key", "dim_year"), (dim_language, "language_key", "dim_language"),
                          (dim_genre, "genre_key", "dim_genre"), (bridge, ["movie_key", "genre_key"], "bridge_movie_genre"),
                          (fact, ["movie_key", "source_key"], "fact_rating")]:
        assert_unique(df, key, f"gold.{name}")
    write_delta(dim_movie, GOLD, "dim_movie", comment="Star schema: dimensão filme (atributos descritivos)")
    write_delta(dim_source, GOLD, "dim_source", comment="Star schema: dimensão fonte da nota")
    write_delta(dim_year, GOLD, "dim_year", comment="Star schema: dimensão ano de estreia e década")
    write_delta(dim_language, GOLD, "dim_language", comment="Star schema: dimensão idioma original")
    write_delta(dim_genre, GOLD, "dim_genre", comment="Star schema: dimensão género (IMDb)")
    write_delta(bridge, GOLD, "bridge_movie_genre", comment="Star schema: ponte filme-género (muitos-para-muitos)")
    write_delta(fact, GOLD, "fact_rating", cluster_by=["source_key", "year_key"],
                comment="Star schema: factos de avaliação, grão = filme x fonte, nota na escala 1-10")
    add_star_constraints()


# [Auxiliar] Desenho do star schema em SVG. O que interessa é a imagem, não o código de desenho.
def mostrar_star_schema():
    """Diagrama do star schema em SVG: factos ao centro, dimensões à volta, com PK, FK e cardinalidades."""
    W_FACT, W_DIM, ROW, HEAD = 230, 200, 21, 30
    # Paleta em tons de verde (o verde base é o da Figura 4)
    VERDE_ESCURO, VERDE, VERDE_CLARO = "#2F6B1A", "#5FA339", "#86B86A"
    VERDE_FUNDO, VERDE_LINHA, VERDE_OLIVA = "#F3F9EE", "#9DC183", "#6B8E23"
    # nome: (x, y, largura, cor, [(coluna, tipo, papel)])  papel: PK, FK, M (medida) ou ""
    tables = {
        "fact_rating": (365, 200, W_FACT, VERDE_ESCURO, [
            ("movie_key", "int", "PK FK"), ("source_key", "int", "PK FK"), ("year_key", "int", "FK"),
            ("language_key", "string", "FK"), ("rating_10", "double", "M"), ("n_votes", "bigint", "M")]),
        "dim_source": (380, 15, W_DIM, VERDE, [
            ("source_key", "int", "PK"), ("source_name", "string", ""), ("escala_original", "string", "")]),
        "dim_year": (730, 80, W_DIM, VERDE, [("year_key", "int", "PK"), ("decade", "int", "")]),
        "dim_language": (730, 330, W_DIM, VERDE, [
            ("language_key", "string", "PK"), ("language_name", "string", "")]),
        "dim_movie": (30, 175, W_DIM, VERDE, [
            ("movie_key", "int", "PK"), ("tconst", "string", ""), ("tmdb_id", "int", ""),
            ("title", "string", ""), ("runtime_minutes", "int", ""), ("budget", "bigint", ""),
            ("revenue", "bigint", ""), ("budget_band", "string", "")]),
        "bridge_movie_genre": (260, 470, W_DIM, VERDE_CLARO, [
            ("movie_key", "int", "PK FK"), ("genre_key", "int", "PK FK")]),
        "dim_genre": (530, 470, W_DIM, VERDE, [("genre_key", "int", "PK"), ("genre_name", "string", "")]),
    }
    edges = [("fact_rating", "dim_source"), ("fact_rating", "dim_year"), ("fact_rating", "dim_language"),
             ("fact_rating", "dim_movie"), ("bridge_movie_genre", "dim_movie"), ("bridge_movie_genre", "dim_genre")]

    def geom(name):
        x, y, w, _, cols = tables[name]
        return x, y, w, HEAD + ROW * len(cols) + 6

    def center(name):
        x, y, w, h = geom(name)
        return x + w / 2, y + h / 2

    def exit_point(name, toward, gap=16):
        """Ponto onde a linha sai da caixa, um pouco para fora (para pôr a cardinalidade)."""
        x, y, w, h = geom(name)
        (cx, cy), (tx, ty) = center(name), toward
        dx, dy = tx - cx, ty - cy
        t = min((w / 2) / abs(dx) if dx else float("inf"), (h / 2) / abs(dy) if dy else float("inf"))
        norm = (dx * dx + dy * dy) ** 0.5
        return cx + dx * t + dx / norm * gap, cy + dy * t + dy / norm * gap

    svg, labels = [], []
    for many, one in edges:   # linhas primeiro, para ficarem por baixo das caixas
        (x1, y1), (x2, y2) = center(many), center(one)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{VERDE_LINHA}" stroke-width="2.5"/>')
        for name, other, label in ((many, (x2, y2), "N"), (one, (x1, y1), "1")):
            lx, ly = exit_point(name, other)
            labels.append(f'<circle cx="{lx}" cy="{ly}" r="10" fill="#fff" stroke="{VERDE}"/>'
                          f'<text x="{lx}" y="{ly + 4}" text-anchor="middle" font-size="11" font-weight="700" '
                          f'fill="{VERDE_ESCURO}">{label}</text>')
    for name, (x, y, w, color, cols) in tables.items():
        _, _, _, h = geom(name)
        is_fact = name.startswith("fact")
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{VERDE_FUNDO}" stroke="{color}" '
                   f'stroke-width="{3 if is_fact else 2}"/>'
                   f'<path d="M{x},{y + 8} a8,8 0 0 1 8,-8 h{w - 16} a8,8 0 0 1 8,8 v{HEAD - 8} h-{w} z" fill="{color}"/>'
                   f'<text x="{x + 10}" y="{y + 20}" font-size="14" font-weight="700" fill="#fff">'
                   f'{"★ " if is_fact else ""}{name}</text>')
        for i, (col, dtype, role) in enumerate(cols):
            ty = y + HEAD + 16 + i * ROW
            weight = "700" if "PK" in role else "400"
            badge_color = {"PK": VERDE_ESCURO, "FK": VERDE, "M": VERDE_OLIVA}
            badges = "".join(f'<tspan fill="{badge_color[r]}" font-weight="700">{r} </tspan>' for r in role.split())
            svg.append(f'<text x="{x + 10}" y="{ty}" font-size="12.5" font-weight="{weight}" fill="#222" '
                       f'font-family="Consolas,monospace">{col}</text>'
                       f'<text x="{x + w - 10}" y="{ty}" font-size="11" text-anchor="end" fill="#777">'
                       f'{badges}{dtype}</text>')
    legend = ('<text x="30" y="585" font-size="12" fill="#444">'
              f'<tspan fill="{VERDE_ESCURO}" font-weight="700">PK</tspan> chave primária · '
              f'<tspan fill="{VERDE}" font-weight="700">FK</tspan> chave estrangeira · '
              f'<tspan fill="{VERDE_OLIVA}" font-weight="700">M</tspan> medida · '
              '1 — N cardinalidade · grão de fact_rating: um filme × uma fonte</text>')
    displayHTML(f"""
    <div style="background:#fff;color:#222;font-family:Segoe UI,Helvetica,Arial,sans-serif;padding:10px">
      <div style="font-size:18px;font-weight:700">Star schema da camada Gold</div>
      <svg viewBox="0 0 960 600" width="100%" style="max-width:960px" xmlns="http://www.w3.org/2000/svg"
           font-family="Segoe UI,Helvetica,Arial,sans-serif">{''.join(svg + labels)}{legend}</svg>
    </div>""")


# O diagrama não depende de dados: aparece sempre nesta secção, qualquer que seja a etapa
mostrar_star_schema()

if correr("gold"):
    gold_star()

    # Exemplo: a mesma pergunta da Figura 4 respondida diretamente sobre o star schema
    display(spark.sql(f"""
        SELECT y.decade, s.source_name,
               COUNT(*)                 AS n_movies,
               ROUND(AVG(f.rating_10), 2) AS avg_rating_10
        FROM {tbl(GOLD, 'fact_rating')} f
        JOIN {tbl(GOLD, 'dim_source')} s ON f.source_key = s.source_key
        JOIN {tbl(GOLD, 'dim_year')}   y ON f.year_key   = y.year_key
        GROUP BY y.decade, s.source_name
        ORDER BY y.decade, s.source_name"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Perguntas descritivas e de benchmarking (comportamento dos utilizadores MovieLens)
# MAGIC
# MAGIC | # | Pergunta | Tabela gold | Figura |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Que géneros de filmes existem no conjunto de dados? | `genre_catalog` | 11 |
# MAGIC | 2 | Quantos filmes há por género? | `genre_catalog` | 11 |
# MAGIC | 3 | Como se distribuem as avaliações por utilizador? | `user_activity` | 12 |
# MAGIC | 4 | Quais são os filmes com a classificação média mais alta? | `top_movies` | 13 |
# MAGIC | 5 | Como mudam, ao longo do tempo, o volume, a média e a distribuição dos ratings? | `ratings_over_time` | 14 |
# MAGIC | 6 | Que utilizadores são sinalizados por picos de atividade, avaliações no mesmo segundo ou concentração em poucas notas? | `user_activity` | 15 |
# MAGIC | 7 | Como muda a sinalização ao excluir as primeiras 24 horas após a primeira avaliação observada? | `flag_sensitivity` | 15 |
# MAGIC | 8 | Que filmes registam as maiores subidas ou descidas de classificação média entre períodos? | `movie_rating_shifts` | 16 |
# MAGIC | 9 | Em quantas avaliações se baseiam essas mudanças? | `movie_rating_shifts` (`n_ratings_total`) | 16 |
# MAGIC | 10 | Que filmes recebem, no mesmo dia, uma concentração de notas muito altas ou muito baixas face ao seu histórico? | `movie_day_anomalies` | 17 |
# MAGIC | 11 | Que proporção dos participantes já tinha sido sinalizada por atividade invulgar? | `anomaly_participants` | 17 |
# MAGIC
# MAGIC **Regras de sinalização** (constantes na célula seguinte, fáceis de ajustar e justificar):
# MAGIC * **Pico de atividade:** um dia com pelo menos `SPIKE_MIN_DAY` avaliações **e** acima do percentil 99 de todos os dias-utilizador.
# MAGIC * **Mesmo segundo:** `SAME_SECOND_MIN` ou mais avaliações com o mesmo timestamp (difícil para um humano).
# MAGIC * **Concentração em poucas notas:** utilizador com pelo menos `CONC_MIN_RATINGS` avaliações em que a nota mais usada representa `CONC_TOP_SHARE` ou mais do total.
# MAGIC * **Sem as primeiras 24 horas:** no MovieLens, quem se regista avalia muitos filmes de seguida para receber recomendações. Repetir a sinalização sem essas 24 horas separa esse comportamento normal de atividade realmente invulgar.
# MAGIC * **Data da sinalização:** primeiro dia de pico, primeiro segundo suspeito, ou dia em que o utilizador atinge `CONC_MIN_RATINGS` avaliações concentradas. Serve para saber se alguém "já tinha sido sinalizado" antes de participar numa anomalia.
# MAGIC * **Mudança entre períodos:** média de cada filme por períodos de `PERIOD_YEARS` anos (mínimo `MIN_RATINGS_PERIOD` avaliações por período), comparada com o período anterior com dados.
# MAGIC * **Concentração no mesmo dia:** dia com pelo menos `DAY_MIN_RATINGS` avaliações de um filme em que a proporção de notas ≥ `EXTREME_HIGH` (ou ≤ `EXTREME_LOW`) supera a do resto do histórico do filme em `SHARE_DIFF` ou mais, com z ≥ `Z_MIN` (3), ou seja, uma diferença muito pouco provável por acaso.

# COMMAND ----------

# ---------------- 4.3 Parâmetros das regras (fáceis de ajustar) ----------------
SPIKE_MIN_DAY      = 50     # avaliações num só dia
SAME_SECOND_MIN    = 3      # avaliações no mesmo segundo
CONC_MIN_RATINGS   = 20     # mínimo de avaliações para avaliar a concentração
CONC_TOP_SHARE     = 0.70   # proporção da nota mais usada
PERIOD_YEARS       = 5      # tamanho de cada período
MIN_RATINGS_PERIOD = 10     # avaliações mínimas de um filme por período
EXTREME_HIGH, EXTREME_LOW = 4.5, 1.0
DAY_MIN_RATINGS    = 5      # avaliações mínimas de um filme num dia
HIST_MIN_RATINGS   = 10     # histórico mínimo do filme fora desse dia
SHARE_DIFF         = 0.40   # diferença mínima de proporção face ao histórico
Z_MIN              = 3      # z mínimo para considerar a diferença "invulgar"
BAYES_PRIOR_N      = 20     # "votos fictícios" da média bayesiana no top de filmes


def ml_ratings_and_movies():
    """Lê da silver as avaliações e os filmes do MovieLens (usados em todas as perguntas 4.3)."""
    require_tables(SILVER, ["ml_ratings", "ml_movies", "movies_unified"], "movielens e integracao")
    ratings = spark.table(tbl(SILVER, "ml_ratings")).select("user_id", "movie_id", "rating", "rated_at")
    movies = spark.table(tbl(SILVER, "ml_movies")).select("movie_id", "title", "year", "genres")
    return ratings, movies


# COMMAND ----------

# --------------- Perguntas 1-2: que géneros existem e quantos filmes têm ---------------
def gold_genre_catalog():
    _, movies = ml_ratings_and_movies()
    # explode: um filme com 3 géneros passa a 3 linhas (filme, género)
    imdb_g = (spark.table(tbl(SILVER, "movies_unified")).select("movie_id", F.explode("genres").alias("genre"))
              .groupBy("genre").agg(F.countDistinct("movie_id").alias("n_movies_imdb")))
    ml_g = (movies.select("movie_id", F.explode("genres").alias("genre"))
            .groupBy("genre").agg(F.countDistinct("movie_id").alias("n_movies_movielens")))
    # join "full": mantém os géneros que só existem numa das duas fontes (com 0 na outra)
    catalog = imdb_g.join(ml_g, "genre", "full").fillna(0, ["n_movies_imdb", "n_movies_movielens"])
    write_delta(catalog, GOLD, "genre_catalog", comment="Géneros existentes e nº de filmes por género (IMDb e MovieLens)")


if correr("gold"):
    gold_genre_catalog()

# COMMAND ----------

# --------------- Perguntas 3, 6 e 7: atividade e sinalização dos utilizadores ---------------
def user_flags(r, suffix):
    """Aplica as três regras de sinalização a um conjunto de avaliações r
    (colunas user_id, movie_id, rating, rated_at). Devolve uma linha por utilizador.
    O 'suffix' é acrescentado ao nome das colunas (ex.: '_sem_24h') para as distinguir."""
    r = r.withColumn("rating_date", F.to_date("rated_at"))
    users = r.groupBy("user_id").agg(F.count("*").alias("n_ratings"))

    # Regra 1, pico de atividade: dias com >= SPIKE_MIN_DAY avaliações e acima do percentil 99
    per_day = r.groupBy("user_id", "rating_date").agg(F.count("*").alias("n_day"))
    p99 = per_day.agg(F.percentile_approx("n_day", 0.99)).first()[0] or 0
    spikes = (per_day.where((F.col("n_day") >= SPIKE_MIN_DAY) & (F.col("n_day") > p99))
              .groupBy("user_id").agg(F.count("*").alias("spike_days"), F.max("n_day").alias("max_ratings_day"),
                                      F.min("rating_date").alias("spike_date")))

    # Regra 2, mesmo segundo: SAME_SECOND_MIN ou mais avaliações com o mesmo timestamp
    same = (r.groupBy("user_id", "rated_at").agg(F.count("*").alias("n_sec"))
            .where(F.col("n_sec") >= SAME_SECOND_MIN)
            .groupBy("user_id").agg(F.count("*").alias("same_second_events"),
                                    F.max("n_sec").alias("max_same_second"),
                                    F.min(F.to_date("rated_at")).alias("same_second_date")))

    # Regra 3, concentração: que proporção das notas do utilizador é a nota que ele mais usa?
    w_user = Window.partitionBy("user_id")   # janela = "todas as linhas do mesmo utilizador"
    conc = (r.groupBy("user_id", "rating").count()
            .withColumn("share", F.col("count") / F.sum("count").over(w_user))
            .groupBy("user_id").agg(F.max("share").alias("top_share"), F.count("*").alias("distinct_values")))
    # data em que o utilizador chega à avaliação nº CONC_MIN_RATINGS (a partir daí a regra 3 aplica-se)
    nth = (r.withColumn("_rn", F.row_number().over(w_user.orderBy("rated_at", "movie_id")))
           .where(F.col("_rn") == CONC_MIN_RATINGS).select("user_id", F.col("rating_date").alias("conc_date")))

    # Juntar tudo (left join: quem não tem picos fica com null nessas colunas)
    out = (users.join(spikes, "user_id", "left").join(same, "user_id", "left")
           .join(conc, "user_id", "left").join(nth, "user_id", "left")
           .withColumn("flag_spike", F.col("spike_days").isNotNull())
           .withColumn("flag_same_second", F.col("same_second_events").isNotNull())
           .withColumn("flag_concentration", (F.col("n_ratings") >= CONC_MIN_RATINGS) & (F.col("top_share") >= CONC_TOP_SHARE))
           .withColumn("flag_any", F.col("flag_spike") | F.col("flag_same_second") | F.col("flag_concentration")))

    # Data da primeira sinalização = a mais antiga das três datas, contando só as regras que dispararam.
    # F.when(...) sem otherwise dá null quando a regra não disparou; F.least ignora os nulls.
    out = out.withColumn("first_flag_date", F.least(F.when(F.col("flag_spike"), F.col("spike_date")),
                                                    F.when(F.col("flag_same_second"), F.col("same_second_date")),
                                                    F.when(F.col("flag_concentration"), F.col("conc_date"))))
    cols = ["n_ratings", "spike_days", "max_ratings_day", "max_same_second", "top_share", "distinct_values",
            "flag_spike", "flag_same_second", "flag_concentration", "flag_any", "first_flag_date"]
    return out.select("user_id", *[F.col(c).alias(c + suffix) for c in cols])


def gold_user_activity():
    ratings, _ = ml_ratings_and_movies()

    # Pergunta 3: estatísticas de cada utilizador
    stats = ratings.groupBy("user_id").agg(
        F.round(F.avg("rating"), 3).alias("avg_rating"), F.round(F.stddev("rating"), 3).alias("sd_rating"),
        F.min("rated_at").alias("first_rating_at"), F.max("rated_at").alias("last_rating_at"),
        F.countDistinct(F.to_date("rated_at")).alias("active_days"))

    # Pergunta 7: as mesmas avaliações sem as primeiras 24 horas de cada utilizador
    after_24h = (ratings.join(stats.select("user_id", "first_rating_at"), "user_id")
                 .where(F.col("rated_at") >= F.col("first_rating_at") + F.expr("INTERVAL 24 HOURS"))
                 .drop("first_rating_at"))

    # Pergunta 6: sinalização com todas as avaliações (sufixo "") e sem as primeiras 24 h
    activity = (stats.join(user_flags(ratings, ""), "user_id")
                .join(user_flags(after_24h, "_sem_24h"), "user_id", "left")
                .fillna(False, ["flag_spike_sem_24h", "flag_same_second_sem_24h",
                                "flag_concentration_sem_24h", "flag_any_sem_24h"]))
    assert_unique(activity, "user_id", "gold.user_activity")
    activity = write_delta(activity, GOLD, "user_activity",
                           comment="Atividade e sinalização por utilizador MovieLens (todas as avaliações e sem as primeiras 24 h)")

    # Pergunta 7 (resumo): quantos utilizadores cada regra sinaliza, com e sem as primeiras 24 h
    flag_cols = [("flag_spike", "Pico de atividade"), ("flag_same_second", "Avaliações no mesmo segundo"),
                 ("flag_concentration", "Concentração em poucas notas"), ("flag_any", "Qualquer sinalização")]
    rows = []
    for col, label in flag_cols:
        todas = activity.where(F.col(col)).count()
        sem_24h = activity.where(F.col(col + "_sem_24h")).count()
        rows.append((label, todas, sem_24h))
    sens_df = (spark.createDataFrame(rows, "sinalizacao string, utilizadores_todas long, utilizadores_sem_24h long")
               .withColumn("total_utilizadores", F.lit(activity.count()))
               .withColumn("variacao_pct", F.round(100 * (F.col("utilizadores_sem_24h") - F.col("utilizadores_todas"))
                                                    / F.greatest(F.col("utilizadores_todas"), F.lit(1)), 1)))
    write_delta(sens_df, GOLD, "flag_sensitivity", comment="Utilizadores sinalizados com e sem as primeiras 24 horas")


if correr("gold"):
    gold_user_activity()

# COMMAND ----------

# --------------- Pergunta 4: filmes com a classificação média mais alta ---------------
def gold_top_movies():
    ratings, movies = ml_ratings_and_movies()
    # Média bayesiana: junta BAYES_PRIOR_N "votos fictícios" com a média global.
    # Um filme com 2 notas de 5 fica perto da média global; um com 500 notas de 4,5 quase não muda.
    global_mean = ratings.agg(F.avg("rating")).first()[0]
    top = (ratings.groupBy("movie_id").agg(F.count("*").alias("n_ratings"), F.avg("rating").alias("avg_rating"))
           .join(movies, "movie_id", "left")
           .withColumn("bayes_rating", (F.col("n_ratings") * F.col("avg_rating") + BAYES_PRIOR_N * F.lit(global_mean))
                                       / (F.col("n_ratings") + BAYES_PRIOR_N))
           .withColumn("rank_bayes", F.row_number().over(Window.orderBy(F.desc("bayes_rating"), F.desc("n_ratings"))))
           .select("rank_bayes", "movie_id", "title", "year", "n_ratings",
                   F.round("avg_rating", 3).alias("avg_rating"), F.round("bayes_rating", 3).alias("bayes_rating")))
    write_delta(top, GOLD, "top_movies", comment="Filmes por nota média MovieLens (simples e bayesiana, que penaliza poucas avaliações)")


if correr("gold"):
    gold_top_movies()

# COMMAND ----------

# --------------- Pergunta 5: evolução mensal do volume, da média e da distribuição ---------------
def gold_ratings_over_time():
    ratings, _ = ml_ratings_and_movies()
    values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # Para cada nota possível, a média de (rating == v) é a proporção de avaliações com essa nota
    pct_cols = [F.round(F.avg((F.col("rating") == v).cast("int")), 4).alias("pct_" + str(v).replace(".", "_"))
                for v in values]
    over_time = (ratings.withColumn("month", F.date_trunc("month", "rated_at").cast("date"))
                 .groupBy("month")
                 .agg(F.count("*").alias("n_ratings"), F.round(F.avg("rating"), 3).alias("avg_rating"),
                      F.round(F.stddev("rating"), 3).alias("sd_rating"),
                      F.countDistinct("user_id").alias("n_users"), *pct_cols))
    write_delta(over_time, GOLD, "ratings_over_time", comment="Volume, média e distribuição das notas por mês")


if correr("gold"):
    gold_ratings_over_time()

# COMMAND ----------

# --------------- Perguntas 8-9: maiores subidas e descidas entre períodos ---------------
def gold_rating_shifts():
    ratings, movies = ml_ratings_and_movies()
    # Período de 5 anos: 1997 -> 1995, 2003 -> 2000 (arredonda o ano para baixo ao múltiplo de 5)
    per_period = (ratings.withColumn("period", (F.floor(F.year("rated_at") / PERIOD_YEARS) * PERIOD_YEARS).cast("int"))
                  .groupBy("movie_id", "period").agg(F.count("*").alias("n"), F.avg("rating").alias("avg"))
                  .where(F.col("n") >= MIN_RATINGS_PERIOD))
    # lag = valor da linha anterior na janela, ou seja, o período anterior do mesmo filme
    w_mov = Window.partitionBy("movie_id").orderBy("period")
    shifts = (per_period
              .withColumn("prev_period", F.lag("period").over(w_mov))
              .withColumn("prev_avg", F.lag("avg").over(w_mov))
              .withColumn("prev_n", F.lag("n").over(w_mov))
              .where(F.col("prev_period").isNotNull())          # o primeiro período não tem com que comparar
              .join(movies.select("movie_id", "title"), "movie_id", "left")
              .select("movie_id", "title",
                      F.concat_ws("-", "prev_period", (F.col("prev_period") + PERIOD_YEARS - 1).cast("string")).alias("periodo_anterior"),
                      F.concat_ws("-", "period", (F.col("period") + PERIOD_YEARS - 1).cast("string")).alias("periodo_atual"),
                      F.round("prev_avg", 3).alias("avg_anterior"), F.round("avg", 3).alias("avg_atual"),
                      F.round(F.col("avg") - F.col("prev_avg"), 3).alias("variacao"),
                      F.col("prev_n").alias("n_anterior"), F.col("n").alias("n_atual"),
                      (F.col("prev_n") + F.col("n")).alias("n_ratings_total")))     # pergunta 9
    write_delta(shifts, GOLD, "movie_rating_shifts",
                comment=f"Variação da nota média de cada filme entre períodos de {PERIOD_YEARS} anos (mín. {MIN_RATINGS_PERIOD} avaliações por período)")


if correr("gold"):
    gold_rating_shifts()

# COMMAND ----------

# --------------- Perguntas 10-11: notas extremas concentradas num só dia ---------------
def z_score_proporcao(share_day, share_hist):
    """Quantos desvios-padrão a proporção do dia está acima da proporção habitual do filme.
    Fórmula do teste a uma proporção: z = (p_dia - p_hist) / raiz(p_hist * (1 - p_hist) / n_dia).
    p_hist é limitado a [0,01; 0,99] para nunca dividir por zero."""
    p = F.greatest(F.least(share_hist, F.lit(0.99)), F.lit(0.01))
    return (share_day - share_hist) / F.sqrt(p * (1 - p) / F.col("n_day"))


def gold_day_anomalies():
    ratings, movies = ml_ratings_and_movies()
    daily = (ratings.withColumn("rating_date", F.to_date("rated_at"))
             .withColumn("is_high", (F.col("rating") >= EXTREME_HIGH).cast("int"))   # 1 se nota muito alta
             .withColumn("is_low", (F.col("rating") <= EXTREME_LOW).cast("int")))    # 1 se nota muito baixa

    # Contagens por filme e dia, e totais por filme
    md = daily.groupBy("movie_id", "rating_date").agg(F.count("*").alias("n_day"), F.sum("is_high").alias("high_day"),
                                                     F.sum("is_low").alias("low_day"))
    tot = daily.groupBy("movie_id").agg(F.count("*").alias("n_tot"), F.sum("is_high").alias("high_tot"),
                                        F.sum("is_low").alias("low_tot"))

    # Pergunta 10: comparar a proporção de notas extremas no dia com a do resto do histórico
    # ("histórico" = todas as avaliações do filme menos as desse dia)
    anomalies = (md.join(tot, "movie_id")
                 .withColumn("n_hist", F.col("n_tot") - F.col("n_day"))
                 .where((F.col("n_day") >= DAY_MIN_RATINGS) & (F.col("n_hist") >= HIST_MIN_RATINGS))
                 .withColumn("share_high_day", F.col("high_day") / F.col("n_day"))
                 .withColumn("share_high_hist", (F.col("high_tot") - F.col("high_day")) / F.col("n_hist"))
                 .withColumn("share_low_day", F.col("low_day") / F.col("n_day"))
                 .withColumn("share_low_hist", (F.col("low_tot") - F.col("low_day")) / F.col("n_hist"))
                 .withColumn("z_high", z_score_proporcao(F.col("share_high_day"), F.col("share_high_hist")))
                 .withColumn("z_low", z_score_proporcao(F.col("share_low_day"), F.col("share_low_hist"))))
    # Anomalia = diferença grande (>= SHARE_DIFF) E pouco provável por acaso (z >= Z_MIN)
    high = (F.col("share_high_day") - F.col("share_high_hist") >= SHARE_DIFF) & (F.col("z_high") >= Z_MIN)
    low = (F.col("share_low_day") - F.col("share_low_hist") >= SHARE_DIFF) & (F.col("z_low") >= Z_MIN)
    anomalies = (anomalies.withColumn("tipo", F.when(high, "notas muito altas").when(low, "notas muito baixas"))
                 .where(F.col("tipo").isNotNull()))

    # Pergunta 11: quem avaliou esse filme nesse dia, e se já estava sinalizado antes desse dia
    flags = spark.table(tbl(GOLD, "user_activity")).select("user_id", "first_flag_date")
    participants = (daily.join(F.broadcast(anomalies.select("movie_id", "rating_date", "tipo")), ["movie_id", "rating_date"])
                    .join(flags, "user_id", "left")
                    .withColumn("ja_sinalizado", F.coalesce(F.col("first_flag_date") < F.col("rating_date"), F.lit(False))))
    per_anomaly = participants.groupBy("movie_id", "rating_date").agg(
        F.countDistinct("user_id").alias("n_participants"),
        F.countDistinct(F.when(F.col("ja_sinalizado"), F.col("user_id"))).alias("n_ja_sinalizados"))
    anomalies_out = (anomalies.join(per_anomaly, ["movie_id", "rating_date"], "left")
                     .join(movies.select("movie_id", "title"), "movie_id", "left")
                     .select("movie_id", "title", "rating_date", "tipo", "n_day", "n_hist",
                             F.round("share_high_day", 3).alias("share_high_day"), F.round("share_high_hist", 3).alias("share_high_hist"),
                             F.round("share_low_day", 3).alias("share_low_day"), F.round("share_low_hist", 3).alias("share_low_hist"),
                             F.round(F.greatest("z_high", "z_low"), 2).alias("z"), "n_participants", "n_ja_sinalizados"))
    write_delta(anomalies_out, GOLD, "movie_day_anomalies",
                comment="Dias com concentração anormal de notas muito altas ou muito baixas num filme, face ao seu histórico")

    # Uma linha por participante: já estava sinalizado antes da sua PRIMEIRA participação?
    part_users = (participants.groupBy("user_id")
                  .agg(F.min("rating_date").alias("primeira_participacao"),
                       F.first("first_flag_date", ignorenulls=True).alias("first_flag_date"),
                       F.countDistinct("movie_id", "rating_date").alias("n_anomalias"))
                  .withColumn("ja_sinalizado", F.col("first_flag_date") < F.col("primeira_participacao"))
                  .fillna(False, ["ja_sinalizado"]))
    write_delta(part_users, GOLD, "anomaly_participants",
                comment="Utilizadores que participaram em anomalias diárias e se já estavam sinalizados antes da primeira participação")


if correr("gold"):
    gold_day_anomalies()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluation: análise e gráficos (CRISP-DM fase 5)
# MAGIC
# MAGIC Todos os gráficos leem tabelas gold (ou a silver integrada já filtrada), por isso o `toPandas()` só traz poucas linhas para o driver. Cada gráfico tem por baixo um resumo numérico gerado automaticamente, para ajudar a escrever as conclusões.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Quantos filmes sobrevivem a cada passo de integração?
# MAGIC Ajuda a explicar o tamanho das amostras e os limites das conclusões.

# COMMAND ----------

# A análise só corre se as tabelas de que precisa existirem; caso contrário explica o que falta
# e salta os gráficos, em vez de dar erro.
ANALISE = correr("analise")
if ANALISE:
    _needed = [(SILVER, "movies_unified"), (SILVER, "imdb_movies")] + [(GOLD, t) for t in GOLD_TABLES]
    _missing = [f"{l}.{t}" for l, t in _needed if not spark.catalog.tableExists(tbl(l, t))]
    if _missing:
        ANALISE = False
        print(f"[análise] faltam {_missing}. Corre primeiro as etapas integracao e gold; os gráficos foram saltados.")

if ANALISE:
    unified, base, base_cmp = gold_bases()
    funnel = [
        ("Filmes IMDb (silver)", spark.table(tbl(SILVER, "imdb_movies")).count()),
        ("Ligados ao MovieLens", unified.count()),
        ("Com avaliações MovieLens", unified.where("ml_num_ratings IS NOT NULL").count()),
        ("Com detalhe TMDB", unified.where("in_tmdb").count()),
        (f"base (≥{MIN_IMDB_VOTES} votos IMDb, ≥{MIN_ML_RATINGS} aval. ML)", base.count()),
        (f"base_cmp (+ ≥{MIN_TMDB_VOTES} votos TMDB)", base_cmp.count()),
    ]
    labels, values = zip(*funnel)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels[::-1], values[::-1], color="#4c72b0")
    ax.set_xscale("log")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values[::-1]], padding=3, fontsize=9)
    ax.set_xlabel("Nº de filmes (escala logarítmica)")
    ax.set_title("Figura 2. Funil de integração das três fontes")
    fig.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 H1: as três fontes dão notas diferentes aos mesmos filmes?

# COMMAND ----------

if ANALISE:
    pdf = base_cmp.select("imdb_rating", "ml_rating_10", "tmdb_vote_average").toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(1, 10.25, 0.25)
    for col, name in [("imdb_rating", "IMDb"), ("ml_rating_10", "MovieLens"), ("tmdb_vote_average", "TMDB")]:
        axes[0].hist(pdf[col].dropna(), bins=bins, alpha=0.45, label=f"{name} (média {pdf[col].mean():.2f})",
                     color=SOURCE_COLORS[name])
    axes[0].set_title("Figura 3a. Distribuição das notas (mesmos filmes)")
    axes[0].set_xlabel("Nota (escala 1-10)")
    axes[0].set_ylabel("Nº de filmes")
    axes[0].legend()

    axes[1].scatter(pdf["ml_rating_10"], pdf["imdb_rating"], s=6, alpha=0.3, color="#555")
    axes[1].plot([1, 10], [1, 10], "r--", lw=1, label="notas iguais (y = x)")
    axes[1].set_xlim(1, 10)
    axes[1].set_ylim(1, 10)
    axes[1].set_xlabel("Nota MovieLens (×2)")
    axes[1].set_ylabel("Nota IMDb")
    axes[1].set_title(f"Figura 3b. IMDb vs MovieLens (r = {pdf['imdb_rating'].corr(pdf['ml_rating_10']):.2f})")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    plt.show()

    diff = pdf["imdb_rating"] - pdf["ml_rating_10"]
    print(f"Filmes comparados: {len(pdf):,}")
    print(f"Média IMDb {pdf['imdb_rating'].mean():.2f} | MovieLens {pdf['ml_rating_10'].mean():.2f} | "
          f"TMDB {pdf['tmdb_vote_average'].mean():.2f}")
    print(f"Diferença média IMDb − MovieLens: {diff.mean():+.2f} pontos "
          f"(IC 95%: {diff.mean() - ci95(diff.std(), len(diff)):+.2f} a {diff.mean() + ci95(diff.std(), len(diff)):+.2f})")
    print("Se o intervalo não contém 0, a diferença é estatisticamente significativa (H1).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 H5: as notas mudam com a década?
# MAGIC As bandas sombreadas são intervalos de confiança a 95%. As barras cinzentas mostram quantos filmes há em cada década: décadas com poucos filmes têm bandas largas.

# COMMAND ----------

if ANALISE:
    dec = spark.table(tbl(GOLD, "ratings_by_decade")).orderBy("decade").toPandas()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax2 = ax.twinx()
    ax2.bar(dec["decade"], dec["n_movies"], width=7, color="#dddddd", zorder=0)
    ax2.set_ylabel("Nº de filmes", color="#888")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for prefix, name in [("imdb", "IMDb"), ("ml", "MovieLens"), ("tmdb", "TMDB")]:
        m, h = dec[f"avg_{prefix}"], ci95(dec[f"sd_{prefix}"], dec["n_movies"])
        ax.plot(dec["decade"], m, marker="o", label=name, color=SOURCE_COLORS[name])
        ax.fill_between(dec["decade"], m - h, m + h, alpha=0.15, color=SOURCE_COLORS[name])
    ax.set_xlabel("Década de estreia")
    ax.set_ylabel("Nota média (1-10)")
    ax.set_title("Figura 4. Nota média por década e fonte (mesmos filmes, IC 95%)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

    old, new = dec[dec["decade"] < 1980], dec[dec["decade"] >= 2000]
    if len(old) and len(new):
        print(f"Nota IMDb média antes de 1980: {np.average(old['avg_imdb'], weights=old['n_movies']):.2f} | "
              f"desde 2000: {np.average(new['avg_imdb'], weights=new['n_movies']):.2f}")
        print("Notas mais altas nos filmes antigos são compatíveis com o efeito de sobrevivência (H5).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 H2: a diferença IMDb − MovieLens depende do género?
# MAGIC Barras à direita de zero: o IMDb dá mais pontos do que o MovieLens. As linhas pretas são intervalos de confiança a 95%: se cruzam o zero, a diferença nesse género não é significativa.

# COMMAND ----------

if ANALISE:
    g = spark.table(tbl(GOLD, "gap_by_genre")).orderBy("avg_gap").toPandas()
    err = ci95(g["sd_gap"], g["n_movies"])
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in g["avg_gap"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(g))))
    ax.barh(g["genre"] + "  (n=" + g["n_movies"].astype(str) + ")", g["avg_gap"], xerr=err,
            color=colors, error_kw={"elinewidth": 1})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Diferença média IMDb − MovieLens (pontos, escala 1-10)")
    ax.set_title("Figura 5. Diferença de nota por género (IC 95%)")
    fig.tight_layout()
    plt.show()
    top, bottom = g.iloc[-1], g.iloc[0]
    print(f"Maior gap positivo: {top['genre']} ({top['avg_gap']:+.2f}) | "
          f"menor: {bottom['genre']} ({bottom['avg_gap']:+.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 H3 e H4: idioma e orçamento

# COMMAND ----------

if ANALISE:
    lang = spark.table(tbl(GOLD, "gap_by_language")).orderBy(F.desc("n_movies")).limit(10).toPandas()
    lang["nome"] = lang["original_language"].map(LANG_NAMES).fillna(lang["original_language"])
    bud = spark.table(tbl(GOLD, "gap_by_budget")).orderBy("budget_band").toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(lang))
    axes[0].bar(x - 0.2, lang["avg_gap_ml"], 0.4, yerr=ci95(lang["sd_gap_ml"], lang["n_movies"]),
                label="IMDb − MovieLens", color=SOURCE_COLORS["MovieLens"])
    axes[0].bar(x + 0.2, lang["avg_gap_tmdb"], 0.4, yerr=ci95(lang["sd_gap_tmdb"], lang["n_movies"]),
                label="IMDb − TMDB", color=SOURCE_COLORS["TMDB"])
    axes[0].set_xticks(x, lang["nome"] + "\n(n=" + lang["n_movies"].astype(str) + ")", rotation=45, ha="right", fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Diferença média (pontos)")
    axes[0].set_title("Figura 6a. Diferença de nota por idioma original (IC 95%)")
    axes[0].legend()

    bud["budget_band"] = bud["budget_band"] + "\n(n=" + bud["n_movies"].astype(str) + ")"
    axes[1].bar(bud["budget_band"], bud["avg_gap"], yerr=ci95(bud["sd_gap"], bud["n_movies"]),
                color="#8172b2", label="diferença média")
    axes[1].plot(bud["budget_band"], bud["avg_abs_gap"], "ko--", label="diferença absoluta média")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Orçamento (USD nominais)")
    axes[1].set_ylabel("Pontos")
    axes[1].set_title("Figura 6b. Diferença IMDb − MovieLens por orçamento")
    axes[1].legend()
    fig.tight_layout()
    plt.show()
    print("H4: se a diferença absoluta média desce com o orçamento, os filmes caros têm notas mais consensuais.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 Modelo: o efeito de cada fator, com os outros constantes
# MAGIC Pontos vermelhos são significativos a 5% (o intervalo não cruza o zero).

# COMMAND ----------

if ANALISE:
    coefs = spark.table(tbl(GOLD, "gap_model_coefficients")).where("variavel != '(constante)'").toPandas()
    coefs = coefs.sort_values("coeficiente")
    r2, n = coefs["r2"].iloc[0], coefs["n"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(coefs) + 1.5))
    ax.errorbar(coefs["coeficiente"], coefs["variavel"], xerr=1.96 * coefs["erro_padrao"], fmt="none",
                ecolor="#999", capsize=3)
    ax.scatter(coefs["coeficiente"], coefs["variavel"],
               color=np.where(coefs["significativo_5pct"], "#c44e52", "#999"), zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Efeito no gap IMDb − MovieLens (pontos)")
    ax.set_title(f"Figura 7. Coeficientes da regressão (n = {n:,}, R² = {r2:.2f})")
    fig.tight_layout()
    plt.show()
    for _, r in coefs[coefs["significativo_5pct"]].iterrows():
        print(f"  {r['variavel']}: {r['coeficiente']:+.3f} pontos (p = {r['p_value']:.3g})")
    print(f"O modelo explica {100 * r2:.0f}% da variação do gap; o resto depende de fatores não observados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.7 Dados não estruturados: que tags aparecem quando as fontes discordam?

# COMMAND ----------

if ANALISE:
    tags = spark.table(tbl(GOLD, "tags_by_gap")).toPandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (grupo, color) in zip(axes, [("IMDb mais generoso", "#c44e52"), ("MovieLens mais generoso", "#4c72b0")]):
        t = tags[tags["grupo"] == grupo].nlargest(12, "n_movies").sort_values("n_movies")
        ax.barh(t["tag"], t["n_movies"], color=color)
        ax.set_title(grupo)
        ax.set_xlabel("Nº de filmes com a tag")
    fig.suptitle("Figura 8. Tags mais frequentes nos filmes com diferença ≥ 1 ponto")
    fig.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.8 Retorno financeiro por género
# MAGIC A barra é a mediana de receita/orçamento; a linha vai do 1.º ao 3.º quartil. A mediana é usada porque a média é dominada por poucos sucessos enormes.

# COMMAND ----------

if ANALISE:
    roi = spark.table(tbl(GOLD, "roi_by_genre")).orderBy("median_roi").toPandas()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(roi))))
    ax.barh(roi["genre"], roi["median_roi"], color="#55a868",
            xerr=[roi["median_roi"] - roi["p25_roi"], roi["p75_roi"] - roi["median_roi"]],
            error_kw={"elinewidth": 1, "ecolor": "#333"})
    ax.axvline(1, color="red", ls="--", lw=1, label="receita = orçamento")
    ax.set_xlabel("Receita / orçamento (mediana, com intervalo interquartil)")
    ax.set_title("Figura 9. Retorno financeiro por género (orçamento ≥ 100k USD)")
    ax.legend()
    fig.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.9 Qualidade dos dados ao longo das execuções

# COMMAND ----------

if ANALISE:
    dq = (spark.table(tbl(GOLD, "dq_results"))
          .withColumn("_rn", F.row_number().over(Window.partitionBy("tabela", "regra").orderBy(F.desc("_checked_at"))))
          .where("_rn = 1").orderBy("pct_violacoes").toPandas())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(dq["regra"] + "  [" + dq["tabela"] + "]", dq["pct_violacoes"],
                   color=np.where(dq["pct_violacoes"] > 5, "#c44e52", "#55a868"))
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in dq["pct_violacoes"]], padding=3, fontsize=8)
    ax.set_xlabel("% de linhas que violam a regra (última verificação)")
    ax.set_title("Figura 10. Controlo de qualidade da silver")
    fig.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.10 Evidências técnicas (Spark e Delta Lake)
# MAGIC
# MAGIC * **Lazy evaluation e plano de execução:** o `explain` mostra o plano otimizado antes de qualquer ação; procurar `BroadcastHashJoin` e `AdaptiveSparkPlan`.
# MAGIC * **Liquid Clustering:** `clusteringColumns` no `DESCRIBE DETAIL`.
# MAGIC * **Delta Lake e time travel:** `DESCRIBE HISTORY` mostra cada escrita; na bronze vê-se um `WRITE` em append por cada lote.

# COMMAND ----------

if ANALISE:
    (spark.table(tbl(SILVER, "movies_unified"))
          .join(F.broadcast(spark.table(tbl(SILVER, "ml_movies")).select("movie_id", F.col("year").alias("ml_year"))),
                "movie_id")
          .where(F.col("decade") == 1990)
          .groupBy("decade").agg(F.avg("imdb_rating").alias("avg_imdb_rating"))
          .explain(mode="formatted"))
    display(spark.sql(f"DESCRIBE DETAIL {tbl(SILVER, 'movies_unified')}")
            .select("name", "format", "numFiles", "sizeInBytes", "clusteringColumns"))
    display(spark.sql(f"DESCRIBE HISTORY {tbl(BRONZE, 'tmdb_movie_details')}")
            .select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11 Organização física das tabelas: Liquid Clustering em vez de particionamento
# MAGIC
# MAGIC **Particionamento tradicional** (`PARTITIONED BY` / `.partitionBy()`) divide a tabela em pastas por valor de uma coluna (ex.: uma pasta por ano). Funciona bem em tabelas muito grandes (centenas de GB ou mais) com colunas de baixa cardinalidade, mas em tabelas pequenas cria **muitos ficheiros pequenos** e torna as leituras mais lentas. Além disso, mudar a coluna de partição obriga a reescrever a tabela inteira.
# MAGIC
# MAGIC **Liquid Clustering** (`CLUSTER BY`), dado na aula #06, **substitui o particionamento e o Z-ORDER** nas tabelas Delta: organiza os ficheiros pelas colunas escolhidas, permite filtrar sem ler ficheiros desnecessários (*data skipping*) e as colunas podem ser mudadas com `ALTER TABLE ... CLUSTER BY` **sem reescrever os dados**. Por isso é a opção usada neste projeto. As duas técnicas não podem ser usadas na mesma tabela.
# MAGIC
# MAGIC | Tabela | Colunas de clustering | Porquê |
# MAGIC |---|---|---|
# MAGIC | `bronze.*` | `_batch_id` | a silver e as auditorias filtram por lote de ingestão |
# MAGIC | `silver.imdb_movies` | `start_year` | filtros e agregações por ano/década |
# MAGIC | `silver.ml_ratings` | `movie_id` | agregação das avaliações por filme |
# MAGIC | `silver.movies_unified` | `decade`, `start_year` | todas as análises gold agrupam por década |
# MAGIC | `gold.movie_rating_gaps` | `start_year` | consultas por período |
# MAGIC
# MAGIC As restantes tabelas gold têm poucas dezenas de linhas: não precisam de organização física.
# MAGIC
# MAGIC Para comparação, o particionamento tradicional seria (não executado, porque as tabelas são pequenas):
# MAGIC ```python
# MAGIC df.write.format("delta").partitionBy("decade").saveAsTable("abd_project.silver.movies_unified")
# MAGIC ```
# MAGIC
# MAGIC A célula seguinte corre `OPTIMIZE`, que aplica o clustering aos dados já escritos, e mostra a organização de cada tabela.

# COMMAND ----------

if ANALISE:
    layout = []
    for _layer, _name in EXPECTED_TABLES:
        _full = tbl(_layer, _name)
        if not spark.catalog.tableExists(_full):
            continue
        _detail = spark.sql(f"DESCRIBE DETAIL {_full}").first().asDict()
        _cluster = _detail.get("clusteringColumns") or []
        if _cluster:
            try:
                spark.sql(f"OPTIMIZE {_full}")   # agrupa os ficheiros pelas colunas de clustering
                _detail = spark.sql(f"DESCRIBE DETAIL {_full}").first().asDict()
            except Exception as e:
                print(f"[aviso] OPTIMIZE {_full}: {type(e).__name__}")
        layout.append((_layer, _name, ", ".join(_cluster) or "-", ", ".join(_detail.get("partitionColumns") or []) or "-",
                       int(_detail.get("numFiles") or 0), round((_detail.get("sizeInBytes") or 0) / 1024 / 1024, 2)))
    display(spark.createDataFrame(layout, "camada string, tabela string, clustering string, particoes string, "
                                          "ficheiros long, tamanho_mb double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.12 Respostas às perguntas descritivas e de benchmarking
# MAGIC Cada figura responde a uma ou mais perguntas da tabela da secção 4.3.

# COMMAND ----------

if ANALISE:
    # Perguntas 1 e 2: géneros e nº de filmes por género
    gc = spark.table(tbl(GOLD, "genre_catalog")).toPandas()
    gc = gc.sort_values("n_movies_movielens")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(gc))))
    y = np.arange(len(gc))
    ax.barh(y - 0.2, gc["n_movies_movielens"], 0.4, label="MovieLens", color=SOURCE_COLORS["MovieLens"])
    ax.barh(y + 0.2, gc["n_movies_imdb"], 0.4, label="IMDb (filmes integrados)", color=SOURCE_COLORS["IMDb"])
    ax.set_yticks(y, gc["genre"])
    ax.set_xlabel("Nº de filmes (um filme pode ter vários géneros)")
    ax.set_title(f"Figura 11. {len(gc)} géneros e nº de filmes por género")
    ax.legend()
    fig.tight_layout()
    plt.show()

# COMMAND ----------

if ANALISE:
    # Pergunta 3: distribuição das avaliações por utilizador
    ua = spark.table(tbl(GOLD, "user_activity")).select("n_ratings", "avg_rating", "active_days").toPandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(ua["n_ratings"], bins=np.logspace(np.log10(max(ua["n_ratings"].min(), 1)),
                                                   np.log10(ua["n_ratings"].max() + 1), 30), color="#4c72b0")
    axes[0].set_xscale("log")
    axes[0].axvline(ua["n_ratings"].median(), color="red", ls="--", label=f"mediana = {ua['n_ratings'].median():.0f}")
    axes[0].set_xlabel("Avaliações por utilizador (escala log)")
    axes[0].set_ylabel("Nº de utilizadores")
    axes[0].set_title("Figura 12a. Quantas avaliações faz cada utilizador")
    axes[0].legend()
    axes[1].hist(ua["avg_rating"], bins=np.arange(0.5, 5.25, 0.125), color=SOURCE_COLORS["MovieLens"])
    axes[1].set_xlabel("Nota média do utilizador (0.5 a 5)")
    axes[1].set_title("Figura 12b. Nota média por utilizador")
    fig.tight_layout()
    plt.show()
    top10 = ua["n_ratings"].sort_values(ascending=False)
    print(f"{len(ua):,} utilizadores | mediana {ua['n_ratings'].median():.0f} avaliações | "
          f"os 10% mais ativos fazem {100 * top10.head(max(1, len(ua) // 10)).sum() / top10.sum():.0f}% das avaliações")

# COMMAND ----------

if ANALISE:
    # Pergunta 4: filmes com a classificação média mais alta
    tm = spark.table(tbl(GOLD, "top_movies")).where("rank_bayes <= 15").orderBy(F.desc("rank_bayes")).toPandas()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(tm["title"].str.slice(0, 45), tm["bayes_rating"], color="#55a868")
    ax.bar_label(bars, labels=[f"{b:.2f}  (n={n})" for b, n in zip(tm["bayes_rating"], tm["n_ratings"])],
                 padding=3, fontsize=8)
    ax.set_xlim(tm["bayes_rating"].min() - 0.3, 5.3)
    ax.set_xlabel("Nota média bayesiana (0.5 a 5)")
    ax.set_title("Figura 13. Top 15 filmes por nota média (bayesiana, penaliza poucas avaliações)")
    fig.tight_layout()
    plt.show()

# COMMAND ----------

if ANALISE:
    # Pergunta 5: volume, média e distribuição ao longo do tempo
    ot = spark.table(tbl(GOLD, "ratings_over_time")).orderBy("month").toPandas()
    ot["month"] = pd.to_datetime(ot["month"])
    pct_cols = [c for c in ot.columns if c.startswith("pct_")]
    yearly = ot.assign(year=ot["month"].dt.year)
    yearly = (yearly[pct_cols].mul(yearly["n_ratings"], axis=0).groupby(yearly["year"]).sum()
              .div(yearly.groupby("year")["n_ratings"].sum(), axis=0))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(ot["month"], ot["n_ratings"], width=25, color="#cccccc", label="avaliações por mês")
    axes[0].set_ylabel("Nº de avaliações")
    ax2 = axes[0].twinx()
    ax2.plot(ot["month"], ot["avg_rating"].rolling(6, min_periods=1).mean(), color="#c44e52", label="média (móvel 6 meses)")
    ax2.set_ylabel("Nota média", color="#c44e52")
    ax2.grid(False)
    axes[0].set_title("Figura 14a. Volume mensal e nota média ao longo do tempo")
    axes[1].stackplot(yearly.index, yearly.T.values, labels=[c.replace("pct_", "").replace("_", ".") for c in pct_cols],
                      colors=plt.cm.RdYlGn(np.linspace(0, 1, len(pct_cols))))
    axes[1].set_ylim(0, 1)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylabel("Proporção das avaliações")
    axes[1].set_title("Figura 14b. Distribuição das notas por ano")
    axes[1].legend(title="Nota", ncol=10, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    plt.show()

# COMMAND ----------

if ANALISE:
    # Perguntas 6 e 7: utilizadores sinalizados, com e sem as primeiras 24 horas
    fs = spark.table(tbl(GOLD, "flag_sensitivity")).toPandas()
    x = np.arange(len(fs))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - 0.2, fs["utilizadores_todas"], 0.4, label="todas as avaliações", color="#c44e52")
    b2 = ax.bar(x + 0.2, fs["utilizadores_sem_24h"], 0.4, label="sem as primeiras 24 h", color="#4c72b0")
    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)
    ax.set_xticks(x, fs["sinalizacao"], fontsize=9)
    ax.set_ylabel("Nº de utilizadores sinalizados")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Figura 15. Sinalização de utilizadores (total: {int(fs['total_utilizadores'].iloc[0]):,})")
    ax.legend()
    fig.tight_layout()
    plt.show()
    display(spark.table(tbl(GOLD, "user_activity")).where("flag_any")
            .select("user_id", "n_ratings", "max_ratings_day", "max_same_second", "top_share",
                    "flag_spike", "flag_same_second", "flag_concentration", "flag_any_sem_24h", "first_flag_date")
            .orderBy(F.desc("n_ratings")))
    print("Se a sinalização cai muito sem as primeiras 24 h, os picos eram sobretudo o registo inicial no MovieLens.")

# COMMAND ----------

if ANALISE:
    # Perguntas 8 e 9: maiores subidas e descidas entre períodos, e em quantas avaliações se baseiam
    sh = spark.table(tbl(GOLD, "movie_rating_shifts"))
    best = pd.concat([sh.orderBy(F.desc("variacao")).limit(8).toPandas(),
                      sh.orderBy("variacao").limit(8).toPandas()]).drop_duplicates("movie_id").sort_values("variacao")
    if len(best):
        labels = (best["title"].str.slice(0, 35) + "  " + best["periodo_anterior"] + " → " + best["periodo_atual"])
        fig, ax = plt.subplots(figsize=(11, 0.4 * len(best) + 1.5))
        bars = ax.barh(labels, best["variacao"], color=np.where(best["variacao"] > 0, "#55a868", "#c44e52"))
        ax.bar_label(bars, labels=[f"n = {a} + {b}" for a, b in zip(best["n_anterior"], best["n_atual"])],
                     padding=3, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(min(best["variacao"].min(), 0) - 0.8, max(best["variacao"].max(), 0) + 0.8)
        ax.set_xlabel("Variação da nota média entre períodos (pontos, escala 0.5-5)")
        ax.set_title(f"Figura 16. Maiores subidas e descidas (períodos de {PERIOD_YEARS} anos; n = avaliações em cada período)")
        fig.tight_layout()
        plt.show()
    else:
        print("Nenhum filme tem avaliações suficientes em dois períodos; baixar MIN_RATINGS_PERIOD ou PERIOD_YEARS.")

# COMMAND ----------

if ANALISE:
    # Perguntas 10 e 11: concentrações no mesmo dia e participantes já sinalizados
    an = spark.table(tbl(GOLD, "movie_day_anomalies")).orderBy(F.desc("z"))
    display(an.limit(20))
    pu = spark.table(tbl(GOLD, "anomaly_participants")).toPandas()
    n_an = an.count()
    if len(pu):
        prev = int(pu["ja_sinalizado"].sum())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        by_type = an.groupBy("tipo").count().toPandas()
        axes[0].bar(by_type["tipo"], by_type["count"], color=["#55a868" if "altas" in t else "#c44e52" for t in by_type["tipo"]])
        axes[0].set_ylabel("Nº de dias-filme anómalos")
        axes[0].set_title(f"Figura 17a. {n_an} concentrações anormais no mesmo dia")
        axes[1].bar(["já sinalizados", "não sinalizados"], [prev, len(pu) - prev], color=["#c44e52", "#999999"])
        axes[1].set_title(f"Figura 17b. Participantes: {100 * prev / len(pu):.1f}% já estavam sinalizados")
        axes[1].set_ylabel("Nº de utilizadores")
        fig.tight_layout()
        plt.show()
        print(f"{len(pu):,} utilizadores participaram em {n_an} anomalias; {prev:,} ({100 * prev / len(pu):.1f}%) "
              "já tinham sido sinalizados por atividade invulgar antes da primeira participação.")
    else:
        print("Não foram encontradas concentrações anormais com os limiares atuais.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deployment: pipeline visual com um Job do Databricks (CRISP-DM fase 6)
# MAGIC
# MAGIC Num projeto real o pipeline não corre à mão. Cada etapa deste notebook pode ser uma **tarefa** de um Job, e o Databricks desenha o grafo e mostra o estado de cada tarefa (verde, vermelho, a correr).
# MAGIC
# MAGIC ```
# MAGIC    tmdb ─────┐
# MAGIC    imdb ─────┼──► integracao ──► gold ──► analise
# MAGIC    movielens ┘
# MAGIC ```
# MAGIC
# MAGIC **Criar o Job (5 minutos, na Free Edition):**
# MAGIC 1. Menu da esquerda **Jobs & Pipelines** → **Create** → **Job**.
# MAGIC 2. Primeira tarefa: *Task name* `tmdb`, *Type* **Notebook**, *Path* = este notebook, *Compute* **Serverless**. Em *Parameters* acrescentar `etapa` = `tmdb`.
# MAGIC 3. **Add task** → `imdb` (parâmetro `etapa` = `imdb`). Em *Depends on* **apagar** a dependência que aparece por omissão, para correr em paralelo com `tmdb`.
# MAGIC 4. Igual para `movielens` (`etapa` = `movielens`, sem dependências).
# MAGIC 5. `integracao` (`etapa` = `integracao`), *Depends on*: `tmdb`, `imdb`, `movielens`.
# MAGIC 6. `gold` (`etapa` = `gold`), *Depends on*: `integracao`.
# MAGIC 7. `analise` (`etapa` = `analise`), *Depends on*: `gold`.
# MAGIC 8. **Run now**. No separador **Runs** clicar na execução para ver o grafo; clicar numa tarefa abre o notebook com os outputs e gráficos dessa etapa.
# MAGIC 9. Opcional: **Schedule** diário, ou **File arrival** no Volume `raw` para correr quando chega um ficheiro novo (ingestão incremental automática).
# MAGIC
# MAGIC Em alternativa, o ficheiro `job_filmes_medallion.yml` (entregue com este notebook) tem a definição do Job em código: no Job, menu **⋮ → Edit as YAML** (ou *View as code*) e colar.
# MAGIC
# MAGIC **Num Job o widget `reiniciar` fica sempre em `nao`.** O notebook recusa um reinício fora de `etapa = tudo`, porque apagaria tabelas enquanto outras tarefas as leem. Como a bronze é incremental e as outras camadas são reconstruídas, o Job pode correr quantas vezes for preciso sem duplicar nem perder dados.

# COMMAND ----------

# Inventário final e estado do pipeline
inventory = []
for _layer in LAYERS:
    for _row in spark.sql(f"SHOW TABLES IN {CATALOG}.{_layer}").collect():
        if not _row.isTemporary:
            inventory.append((_layer, _row.tableName, spark.table(tbl(_layer, _row.tableName)).count()))
if inventory:
    display(spark.createDataFrame(inventory, "camada string, tabela string, linhas long").orderBy("camada", "tabela"))
mostrar_pipeline("Estado do pipeline depois de correr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusões
# MAGIC
# MAGIC _(Preencher com os números impressos por baixo de cada figura.)_
# MAGIC
# MAGIC **Sobre a pergunta de negócio**
# MAGIC * **H1 (fontes diferem):** _ex.: "nos mesmos N filmes, o IMDb dá em média +X pontos face ao MovieLens (IC 95% a..b), e o TMDB fica entre os dois"._ (Fig. 3)
# MAGIC * **H2 (género):** _géneros com maior e menor gap, e se são significativos._ (Fig. 5 e 7)
# MAGIC * **H3 (idioma):** _efeito do idioma não inglês no modelo._ (Fig. 6a e 7)
# MAGIC * **H4 (orçamento):** _a diferença absoluta desce ou não com o orçamento._ (Fig. 6b)
# MAGIC * **H5 (década):** _notas mais altas nos filmes antigos e porquê (sobrevivência)._ (Fig. 4)
# MAGIC * **Texto livre:** _que tags caracterizam os filmes onde as fontes discordam._ (Fig. 8)
# MAGIC * **Recomendação para a plataforma:** _ex.: mostrar a nota ajustada por fonte, ou a média das três quando divergem menos de 0,5 pontos._
# MAGIC
# MAGIC **Sobre o processo e a tecnologia**
# MAGIC * Lakehouse em Unity Catalog com três camadas; bronze incremental com histórico de leituras, silver reconstruível a partir da bronze, gold orientada às hipóteses.
# MAGIC * Spark: schemas explícitos, `try_cast`, broadcast joins, AQE, Liquid Clustering, Delta time travel.
# MAGIC * Pipeline executável como Job com tarefas paralelas por fonte.
# MAGIC
# MAGIC **Limitações:** o MovieLens-small cobre ~9 mil filmes e utilizadores que não representam o público geral; o TMDB só tem detalhe para os filmes descarregados; valores de orçamento em USD nominais (sem inflação); a regressão mostra associações, não causalidade.
# MAGIC
# MAGIC **Trabalho futuro:** MovieLens 25M, `title.principals` do IMDb (realizadores e elenco), TF-IDF das sinopses como features, ajuste de inflação, e Auto Loader para ingestão contínua.