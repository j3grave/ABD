"""Silver · limpeza, tipagem e deduplicação (materialized views).

Replica a camada silver do notebook movies_medallion_project_v3: cada tabela é reescrita a
partir de todo o histórico da bronze, ficando com a versão mais recente de cada chave
(`latest`). Aqui isso é uma materialized view: o pipeline recalcula-a quando a bronze muda.
As regras de qualidade do notebook (chaves únicas, intervalos de notas) passam a
expectations do pipeline.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

YEAR_PATTERN = r"\((\d{4})\)\s*$"

# ------------------------- funções auxiliares ---------------------------------

def latest(df, keys, order_cols=("_file_modified",)):
    """Fica só com a versão mais recente de cada chave (determinístico)."""
    w = Window.partitionBy(*keys).orderBy(*[F.col(c).desc() for c in order_cols])
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def nullify(df, tokens=("\\N",), cols=None):
    """Converte marcadores de 'em falta' (ex.: \\N do IMDb) em null, num único select."""
    target = set(cols or [c for c, t in df.dtypes if t == "string"])
    return df.select(*[
        F.when(F.col(c).isin(*tokens), F.lit(None)).otherwise(F.col(c)).alias(c)
        if c in target else F.col(c)
        for c in df.columns
    ])


def try_cast(col_name, dtype):
    """Cast tolerante: valores inválidos ficam null em vez de parar o pipeline."""
    return F.expr(f"try_cast(`{col_name}` AS {dtype})")


def safe_date(col_name):
    """'yyyy-MM-dd' para data; qualquer outro valor fica null."""
    return F.when(F.col(col_name).rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(F.col(col_name)))


def imdb_tconst(col_name):
    """Constrói o tconst do IMDb (tt0114709). Preenche com zeros até 7 dígitos mas
    NUNCA corta: os filmes recentes já têm 8 dígitos (tt10872600)."""
    c = F.trim(F.col(col_name))
    return (F.when(c.isNull() | (c == ""), F.lit(None))
            .otherwise(F.concat(F.lit("tt"),
                               F.when(F.length(c) < 7, F.lpad(c, 7, "0")).otherwise(c))))


# ----------------------------- SILVER · TMDB ---------------------------------
LIST_MOVIE_DDL = ("id long, title string, original_title string, original_language string, overview string, "
                  "release_date string, popularity double, vote_average double, vote_count long")
DETAIL_DDL = ("id long, imdb_id string, title string, original_title string, original_language string, "
              "overview string, release_date string, runtime int, budget long, revenue long, status string, "
              "popularity double, vote_average double, vote_count long, "
              "genres array<struct<id:int,name:string>>")


@dp.materialized_view(
    name="silver_tmdb_movies",
    comment="TMDB: filmes das listas, um por tmdb_id, com as listas em que aparecem",
)
@dp.expect("tmdb_id presente", "tmdb_id IS NOT NULL")
def silver_tmdb_movies():
    lists = (spark.read.table("bronze_tmdb_movie_lists")
             .select("list_name", "_file_modified", F.from_json("payload", LIST_MOVIE_DDL).alias("m"))
             .select("list_name", "_file_modified", F.col("m.id").alias("tmdb_id"), "m.*")
             .where(F.col("tmdb_id").isNotNull()))
    lists_per_movie = lists.groupBy("tmdb_id").agg(F.sort_array(F.collect_set("list_name")).alias("lists"))
    return (latest(lists, ["tmdb_id"])
            .join(lists_per_movie, "tmdb_id")
            .select("tmdb_id", "title", "original_title", "original_language", "overview",
                    safe_date("release_date").alias("release_date"),
                    "popularity", "vote_average", "vote_count", "lists"))


@dp.materialized_view(
    name="silver_tmdb_movie_details",
    comment="TMDB: detalhe tipado (orçamento, receita, duração, géneros), versão mais recente",
)
@dp.expect("tmdb_id presente", "tmdb_id IS NOT NULL")
def silver_tmdb_movie_details():
    details = (spark.read.table("bronze_tmdb_movie_details")
               .select("_file_modified", F.from_json("payload", DETAIL_DDL).alias("d"))
               .select("_file_modified", "d.*")
               .where(F.col("id").isNotNull()))
    return (latest(nullify(details, tokens=("",)), ["id"])
            .select(F.col("id").alias("tmdb_id"), "imdb_id", "title", "original_title",
                    "original_language", "overview", safe_date("release_date").alias("release_date"),
                    F.col("runtime").alias("runtime_minutes"), "budget", "revenue", "status",
                    "popularity", "vote_average", "vote_count", "genres"))


@dp.materialized_view(
    name="silver_tmdb_movie_genres",
    comment="TMDB: relação filme-género",
)
def silver_tmdb_movie_genres():
    return (spark.read.table("silver_tmdb_movie_details")
            .select("tmdb_id", F.explode("genres").alias("g"))
            .select("tmdb_id", F.col("g.id").alias("genre_id"), F.col("g.name").alias("genre_name")))


# ----------------------------- SILVER · IMDb ---------------------------------

@dp.materialized_view(
    name="silver_imdb_movies",
    cluster_by=["start_year"],
    comment="IMDb: filmes não adultos, tipados, com nota e votos (Liquid Clustering por start_year)",
)
@dp.expect("nota IMDb entre 1 e 10", "imdb_rating IS NULL OR imdb_rating BETWEEN 1 AND 10")
@dp.expect("ano de estreia presente", "start_year IS NOT NULL")
def silver_imdb_movies():
    basics = latest(nullify(spark.read.table("bronze_imdb_title_basics")), ["tconst"])
    ratings = latest(nullify(spark.read.table("bronze_imdb_title_ratings")), ["tconst"])

    imdb_ratings = ratings.select(
        "tconst",
        try_cast("averageRating", "double").alias("imdb_rating"),
        try_cast("numVotes", "bigint").alias("imdb_votes"))
    return (basics.where(F.col("titleType") == "movie")
            .select("tconst",
                    F.col("primaryTitle").alias("title"),
                    F.col("originalTitle").alias("original_title"),
                    F.coalesce(F.col("isAdult") == "1", F.lit(False)).alias("is_adult"),
                    try_cast("startYear", "int").alias("start_year"),
                    try_cast("runtimeMinutes", "int").alias("runtime_minutes"),
                    F.split("genres", ",").alias("genres"))
            .where(~F.col("is_adult"))
            .join(imdb_ratings, "tconst", "left"))


@dp.materialized_view(
    name="silver_imdb_movie_genres",
    comment="IMDb: relação filme-género (nomes em inglês)",
)
def silver_imdb_movie_genres():
    return spark.read.table("silver_imdb_movies").select("tconst", F.explode("genres").alias("genre"))


# --------------------------- SILVER · MovieLens -------------------------------

@dp.materialized_view(
    name="silver_ml_ratings",
    cluster_by=["movie_id"],
    comment="MovieLens: uma avaliação por utilizador e filme (escala 0.5 a 5)",
)
@dp.expect("nota MovieLens entre 0.5 e 5", "rating BETWEEN 0.5 AND 5")
def silver_ml_ratings():
    ml_ratings = (spark.read.table("bronze_ml_ratings")
                  .select(try_cast("userId", "int").alias("user_id"),
                          try_cast("movieId", "int").alias("movie_id"),
                          try_cast("rating", "double").alias("rating"),
                          F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("rated_at"),
                          "_file_modified")
                  .where(F.col("user_id").isNotNull() & F.col("movie_id").isNotNull()
                         & F.col("rating").between(0.5, 5.0)))
    # Uma avaliação por (utilizador, filme): se houver repetidas fica a mais recente.
    return (latest(ml_ratings, ["user_id", "movie_id"], ("_file_modified", "rated_at"))
            .drop("_file_modified"))


@dp.materialized_view(
    name="silver_ml_movies",
    comment="MovieLens: filmes com título, ano e géneros em array",
)
def silver_ml_movies():
    return (latest(spark.read.table("bronze_ml_movies"), ["movieId"])
            .select(try_cast("movieId", "int").alias("movie_id"),
                    F.trim(F.regexp_replace("title", YEAR_PATTERN, "")).alias("title"),
                    F.when(F.regexp_extract("title", YEAR_PATTERN, 1) != "",
                           F.regexp_extract("title", YEAR_PATTERN, 1).cast("int")).alias("year"),
                    F.filter(F.split("genres", r"\|"),
                             lambda g: (g != "") & (g != "(no genres listed)")).alias("genres"))
            .where(F.col("movie_id").isNotNull()))


@dp.materialized_view(
    name="silver_ml_links",
    comment="MovieLens: ponte movie_id <-> tconst (IMDb) <-> tmdb_id (TMDB)",
)
def silver_ml_links():
    return (latest(spark.read.table("bronze_ml_links"), ["movieId"])
            .select(try_cast("movieId", "int").alias("movie_id"),
                    imdb_tconst("imdbId").alias("tconst"),          # chave do IMDb (sem cortar 8 dígitos)
                    try_cast("tmdbId", "int").alias("tmdb_id"))     # chave do TMDB
            .where(F.col("movie_id").isNotNull()))


@dp.materialized_view(
    name="silver_ml_tags",
    comment="MovieLens: tags de texto livre normalizadas (minúsculas)",
)
def silver_ml_tags():
    return (spark.read.table("bronze_ml_tags")
            .select(try_cast("userId", "int").alias("user_id"),
                    try_cast("movieId", "int").alias("movie_id"),
                    F.lower(F.trim("tag")).alias("tag"),
                    F.timestamp_seconds(try_cast("timestamp", "bigint")).alias("tagged_at"))
            .where(F.col("tag").isNotNull() & (F.col("tag") != ""))
            .dropDuplicates(["user_id", "movie_id", "tag"]))


# -------------------------- SILVER INTEGRADA ----------------------------------
# Junta as três fontes por chave. Valores 0 de orçamento, receita e votos do TMDB
# significam "desconhecido" e passam a null. A nota do MovieLens (0.5 a 5) é
# multiplicada por 2 para ficar na escala do IMDb (1 a 10).

@dp.materialized_view(
    name="silver_movies_unified",
    cluster_by=["decade", "start_year"],
    comment="Filmes integrados (IMDb + MovieLens + TMDB); ml_rating_10 = nota MovieLens na escala 1-10",
)
def silver_movies_unified():
    imdb = spark.read.table("silver_imdb_movies")
    links = spark.read.table("silver_ml_links").where(F.col("tconst").isNotNull())
    tmdb = spark.read.table("silver_tmdb_movie_details")

    ml_stats = (spark.read.table("silver_ml_ratings")
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

    return (imdb.join(F.broadcast(links), "tconst", "inner")        # IMDb (grande) x ponte MovieLens (pequena)
            .join(F.broadcast(tmdb_sel), "tmdb_id", "left")     # + TMDB
            .join(F.broadcast(ml_stats), "movie_id", "left")    # + avaliações MovieLens
            .select("movie_id", "tconst", "tmdb_id", "title", "start_year",
                    (F.floor(F.col("start_year") / 10) * 10).cast("int").alias("decade"),
                    "runtime_minutes", "genres", "imdb_rating", "imdb_votes",
                    F.round("ml_avg_rating", 3).alias("ml_avg_rating"),
                    F.round(F.col("ml_avg_rating") * 2, 3).alias("ml_rating_10"), "ml_num_ratings",
                    "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity",
                    "budget", "revenue", "original_language", "overview",
                    F.coalesce(F.col("in_tmdb"), F.lit(False)).alias("in_tmdb")))
