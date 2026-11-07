"""Bronze · ingestão incremental com Auto Loader.

Replica a camada bronze do notebook movies_medallion_project_v3: os ficheiros brutos de
/Volumes/abd_project/bronze/raw entram tal como vieram (CSV/TSV tudo em string; no JSON o
payload guarda o objeto completo como texto), mantendo o histórico de leituras. O Auto
Loader regista os ficheiros já processados: um ficheiro novo carregado para o Volume
entra exatamente uma vez, sem duplicados.

Colunas de auditoria em todas as tabelas: _source, _source_file, _file_modified.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

RAW_ROOT = "/Volumes/abd_project/bronze/raw"


def _ingest(source, file_glob, fmt, schema, **options):
    """Lê uma fonte do Volume com Auto Loader e acrescenta as colunas de auditoria."""
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", fmt)
        .option("pathGlobFilter", file_glob)
    )
    for key, value in options.items():
        reader = reader.option(key, value)
    df = reader.schema(schema).load(f"{RAW_ROOT}/{source}")
    return (
        df.withColumn("_source", F.lit(source))
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_file_modified", F.col("_metadata.file_modification_time"))
    )


# ----------------------------- BRONZE · TMDB ---------------------------------
# O 'payload' guarda o JSON completo como texto (componente não estruturada).

@dp.table(
    name="bronze_tmdb_movie_lists",
    comment="TMDB: listas popular/top_rated/now_playing (JSON bruto, histórico de leituras)",
)
def bronze_tmdb_movie_lists():
    return _ingest("tmdb", "movie_lists*.jsonl", "json",
                   "list_name string, page int, payload string")


@dp.table(
    name="bronze_tmdb_genres",
    comment="TMDB: lista de géneros (JSON bruto)",
)
def bronze_tmdb_genres():
    return _ingest("tmdb", "genres*.jsonl", "json", "payload string")


@dp.table(
    name="bronze_tmdb_movie_details",
    comment="TMDB: detalhe por filme (JSON bruto, histórico de leituras)",
)
def bronze_tmdb_movie_details():
    return _ingest("tmdb", "movie_details*.jsonl", "json",
                   "movie_id long, payload string")


# ----------------------------- BRONZE · IMDb ---------------------------------
# TSV do IMDb: separador tab e sem aspas (os títulos podem conter aspas).

@dp.table(
    name="bronze_imdb_title_basics",
    comment="IMDb title.basics (só filmes), valores brutos em string",
)
def bronze_imdb_title_basics():
    return _ingest(
        "imdb", "title.basics.movies*.tsv.gz", "csv",
        "tconst string, titleType string, primaryTitle string, originalTitle string, "
        "isAdult string, startYear string, endYear string, runtimeMinutes string, genres string",
        sep="\t", quote="", header=True,
    )


@dp.table(
    name="bronze_imdb_title_ratings",
    comment="IMDb title.ratings (só filmes), valores brutos em string",
)
def bronze_imdb_title_ratings():
    return _ingest(
        "imdb", "title.ratings.movies*.tsv.gz", "csv",
        "tconst string, averageRating string, numVotes string",
        sep="\t", quote="", header=True,
    )


# --------------------------- BRONZE · MovieLens -------------------------------
# CSV com aspas duplicadas dentro dos títulos e tags, daí escape='"'.

@dp.table(name="bronze_ml_ratings", comment="MovieLens ratings.csv, valores brutos em string")
def bronze_ml_ratings():
    return _ingest("movielens", "ratings*.csv", "csv",
                   "userId string, movieId string, rating string, timestamp string",
                   escape='"', header=True)


@dp.table(name="bronze_ml_movies", comment="MovieLens movies.csv, valores brutos em string")
def bronze_ml_movies():
    return _ingest("movielens", "movies*.csv", "csv",
                   "movieId string, title string, genres string",
                   escape='"', multiLine=True, header=True)


@dp.table(name="bronze_ml_links", comment="MovieLens links.csv, valores brutos em string")
def bronze_ml_links():
    return _ingest("movielens", "links*.csv", "csv",
                   "movieId string, imdbId string, tmdbId string",
                   escape='"', header=True)


@dp.table(name="bronze_ml_tags", comment="MovieLens tags.csv, valores brutos em string")
def bronze_ml_tags():
    return _ingest("movielens", "tags*.csv", "csv",
                   "userId string, movieId string, tag string, timestamp string",
                   escape='"', multiLine=True, header=True)
