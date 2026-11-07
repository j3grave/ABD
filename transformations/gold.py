"""Gold · tabelas de análise (materialized views).

Replica a camada gold do notebook movies_medallion_project_v3. Os limiares mínimos de
votos são parâmetros do pipeline (min_imdb_votes, min_ml_ratings, min_tmdb_votes),
definidos nas configurações do pipeline — equivalentes aos widgets do notebook.
O modelo de regressão e os gráficos (análise CRISP-DM fase 5/6) continuam no notebook,
porque usam pandas/numpy e não são datasets declarativos.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

MIN_IMDB_VOTES = int(spark.conf.get("min_imdb_votes", "1000"))
MIN_ML_RATINGS = int(spark.conf.get("min_ml_ratings", "10"))
MIN_TMDB_VOTES = int(spark.conf.get("min_tmdb_votes", "20"))


def _stats(col, prefix):
    """Média, desvio-padrão e nº de valores de uma coluna."""
    return [F.round(F.avg(col), 3).alias(f"avg_{prefix}"),
            F.round(F.stddev(col), 3).alias(f"sd_{prefix}")]


def _base():
    """base: filmes com nota IMDb e MovieLens fiáveis; base_cmp: subconjunto também com TMDB."""
    unified = spark.read.table("silver_movies_unified")
    base = (unified
           .where((F.col("imdb_votes") >= MIN_IMDB_VOTES)
                  & (F.col("ml_num_ratings") >= MIN_ML_RATINGS)
                  & F.col("decade").isNotNull())
           .withColumn("gap_imdb_ml", F.col("imdb_rating") - F.col("ml_rating_10")))
    base_cmp = (base.where(F.col("tmdb_vote_count") >= MIN_TMDB_VOTES)
                .withColumn("gap_imdb_tmdb", F.col("imdb_rating") - F.col("tmdb_vote_average")))
    return unified, base, base_cmp


@dp.materialized_view(
    name="gold_ratings_by_decade",
    comment="Nota média e desvio-padrão por fonte e década, nos mesmos filmes (base_cmp)",
)
def gold_ratings_by_decade():
    _, _, base_cmp = _base()
    return (base_cmp.groupBy("decade")
            .agg(F.count("*").alias("n_movies"),
                 *_stats("imdb_rating", "imdb"), *_stats("ml_rating_10", "ml"),
                 *_stats("tmdb_vote_average", "tmdb"))
            .where("n_movies >= 5"))


@dp.materialized_view(
    name="gold_gap_by_genre",
    comment="Diferença IMDb - MovieLens por género (base, mín. 20 filmes)",
)
def gold_gap_by_genre():
    _, base, _ = _base()
    return (base.withColumn("genre", F.explode("genres"))
            .groupBy("genre")
            .agg(F.count("*").alias("n_movies"), *_stats("gap_imdb_ml", "gap"),
                 F.round(F.avg("imdb_rating"), 3).alias("avg_imdb"),
                 F.round(F.avg("ml_rating_10"), 3).alias("avg_ml"))
            .where("n_movies >= 20"))


@dp.materialized_view(
    name="gold_gap_by_language",
    comment="Diferenças de nota por idioma original (base_cmp, mín. 10 filmes)",
)
def gold_gap_by_language():
    _, _, base_cmp = _base()
    return (base_cmp.groupBy("original_language")
            .agg(F.count("*").alias("n_movies"),
                 *_stats("gap_imdb_ml", "gap_ml"), *_stats("gap_imdb_tmdb", "gap_tmdb"))
            .where("n_movies >= 10"))


@dp.materialized_view(
    name="gold_gap_by_budget",
    comment="Diferença IMDb - MovieLens por escalão de orçamento (USD nominais)",
)
def gold_gap_by_budget():
    _, base, _ = _base()
    return (base.where(F.col("budget") >= 100_000)
            .withColumn("budget_band",
                        F.when(F.col("budget") < 5e6, "1) < 5M")
                         .when(F.col("budget") < 30e6, "2) 5-30M")
                         .when(F.col("budget") < 100e6, "3) 30-100M")
                         .otherwise("4) >= 100M"))
            .groupBy("budget_band")
            .agg(F.count("*").alias("n_movies"), *_stats("gap_imdb_ml", "gap"),
                 F.round(F.avg(F.abs("gap_imdb_ml")), 3).alias("avg_abs_gap")))


@dp.materialized_view(
    name="gold_movie_rating_gaps",
    cluster_by=["start_year"],
    comment="Diferença entre a nota IMDb e a nota MovieLens (escala 1-10) por filme",
)
def gold_movie_rating_gaps():
    _, base, _ = _base()
    return base.select("movie_id", "tconst", "title", "start_year", "decade", "genres",
                       "imdb_rating", "imdb_votes", "ml_rating_10", "ml_num_ratings",
                       F.round("gap_imdb_ml", 2).alias("gap_imdb_ml"),
                       F.round(F.abs("gap_imdb_ml"), 2).alias("abs_gap"))


@dp.materialized_view(
    name="gold_roi_by_genre",
    comment="Retorno (receita/orçamento) por género: mediana e quartis, orçamento >= 100k USD",
)
def gold_roi_by_genre():
    unified, _, _ = _base()
    return (unified.where((F.col("budget") >= 100_000) & F.col("revenue").isNotNull())
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


@dp.materialized_view(
    name="gold_tags_by_gap",
    comment="Tags MovieLens mais frequentes nos filmes onde IMDb e MovieLens divergem >= 1 ponto",
)
def gold_tags_by_gap():
    groups = (spark.read.table("gold_movie_rating_gaps")
              .withColumn("grupo", F.when(F.col("gap_imdb_ml") >= 1, "IMDb mais generoso")
                                    .when(F.col("gap_imdb_ml") <= -1, "MovieLens mais generoso"))
              .where(F.col("grupo").isNotNull())
              .select("movie_id", "grupo"))
    return (spark.read.table("silver_ml_tags").join(F.broadcast(groups), "movie_id")
            .groupBy("grupo", "tag").agg(F.countDistinct("movie_id").alias("n_movies"))
            .where("n_movies >= 3"))
