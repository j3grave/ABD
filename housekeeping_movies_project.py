# Databricks notebook source
# MAGIC %md
# MAGIC # Limpeza do projeto dos filmes (apagar tudo o que o projeto criou)
# MAGIC
# MAGIC Este notebook **desfaz** o projeto `movies_medallion_project_v3`: apaga as **tabelas**, os **dados** (ficheiros do Volume `raw`), os **schemas**, o **catálogo**, o **Job** `filmes_medallion` e os **ficheiros do projeto no Workspace**.
# MAGIC
# MAGIC ### Como usar (3 passos)
# MAGIC
# MAGIC 1. **Run all** com os widgets como estão. Corre em **modo simulação**: mostra uma tabela com tudo o que *seria* apagado e **não apaga nada**.
# MAGIC 2. Ler a tabela com calma. Cada linha diz `APAGAR` ou `MANTER` e porquê.
# MAGIC 3. Se estiver tudo certo: widget `2. Modo` = `apagar`, escrever no widget `3. Confirmação` exatamente **`APAGAR abd_project`** (ou `APAGAR <nome do catálogo>`) e fazer **Run all** outra vez.
# MAGIC
# MAGIC Se o modo for `apagar` mas a confirmação estiver errada, o notebook **continua em simulação** (não apaga nada).
# MAGIC
# MAGIC ### O que este notebook NUNCA apaga
# MAGIC
# MAGIC | Objeto | Porquê |
# MAGIC |---|---|
# MAGIC | Outros catálogos, e os catálogos `workspace`, `main`, `system`, `samples`, `hive_metastore` | só se mexe no catálogo do widget; os do sistema estão protegidos |
# MAGIC | Tabelas, volumes ou schemas que **não** foram criados pelo projeto | os nomes vêm do próprio notebook do projeto; o resto fica marcado `MANTER` |
# MAGIC | O catálogo, se tiver outros schemas para além dos do projeto | apagar o catálogo apagaria trabalho de outras aulas |
# MAGIC | Jobs criados por outra pessoa, ou com outro nome | só o Job `filmes_medallion` criado por ti |
# MAGIC | Ficheiros do Workspace fora da tua pasta pessoal (`/Users/<o-teu-email>`) e a própria pasta pessoal | evita apagar o Workspace inteiro |
# MAGIC | Este notebook de limpeza (e a pasta onde está) | não se pode apagar a si próprio enquanto corre |
# MAGIC
# MAGIC ### ⚠️ Antes de apagar
# MAGIC
# MAGIC * **Guarda uma cópia do notebook do projeto** (File → Export → Source file). Depois de apagado no Workspace, só fica a cópia que tiveres fora do Databricks.
# MAGIC * Os **ficheiros do Volume `raw`** (IMDb, TMDB, MovieLens) desaparecem para sempre. Se os quiseres guardar, descarrega-os antes ou põe o widget `4. Apagar dados do Volume raw` = `nao` (nesse caso o schema `bronze` e o Volume ficam).
# MAGIC * As tabelas geridas (*managed*) do Unity Catalog podem ser recuperadas durante 7 dias com `UNDROP TABLE`, mas **só se o schema ainda existir**. Depois de apagar o schema e o catálogo, conta que não há volta atrás.
# MAGIC
# MAGIC ### Ordem da limpeza
# MAGIC
# MAGIC 1. **Job** primeiro, para não voltar a correr e recriar tabelas a meio da limpeza.
# MAGIC 2. **Tabelas** (gold → silver → backup → bronze). Na gold, a tabela de factos vai antes das dimensões, por causa das chaves estrangeiras.
# MAGIC 3. **Volume `raw`** com os ficheiros (os dados brutos).
# MAGIC 4. **Schemas** `gold`, `silver`, `backup`, `bronze`, só se estiverem vazios.
# MAGIC 5. **Catálogo**, só se depois disto só restar o schema `default` vazio.
# MAGIC 6. **Ficheiros do Workspace** (notebook do projeto, YAML do Job, imagens).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parâmetros e regras de segurança

# COMMAND ----------

import re

import pandas as pd

dbutils.widgets.text("catalog", "abd_project", "1. Catálogo do projeto")
dbutils.widgets.dropdown("modo", "simulacao", ["simulacao", "apagar"], "2. Modo")
dbutils.widgets.text("confirmacao", "", "3. Confirmação (escrever APAGAR <catálogo>)")
dbutils.widgets.dropdown("apagar_dados_raw", "sim", ["sim", "nao"], "4. Apagar dados do Volume raw")
dbutils.widgets.dropdown("apagar_catalogo", "sim", ["sim", "nao"], "5. Apagar o catálogo no fim")
dbutils.widgets.text("nome_job", "filmes_medallion", "6. Nome do Job")
dbutils.widgets.text("caminhos_extra", "", "7. Pastas/ficheiros extra no Workspace (separar por vírgulas)")

CATALOG          = dbutils.widgets.get("catalog").strip()
MODO             = dbutils.widgets.get("modo")
CONFIRMACAO      = dbutils.widgets.get("confirmacao").strip()
APAGAR_DADOS_RAW = dbutils.widgets.get("apagar_dados_raw") == "sim"
APAGAR_CATALOGO  = dbutils.widgets.get("apagar_catalogo") == "sim"
NOME_JOB         = dbutils.widgets.get("nome_job").strip()
CAMINHOS_EXTRA   = [c.strip() for c in dbutils.widgets.get("caminhos_extra").split(",") if c.strip()]

# Um nome de catálogo só pode ter letras, números e "_": impede que se escreva SQL no widget.
if not re.fullmatch(r"[A-Za-z0-9_]+", CATALOG):
    raise ValueError(f"Nome de catálogo inválido: '{CATALOG}'. Usa só letras, números e _.")

# Catálogos que nunca são apagados (podemos limpar os schemas do projeto lá dentro, mas o catálogo fica).
CATALOGOS_PROTEGIDOS = {"workspace", "main", "system", "samples", "hive_metastore"}

# Só apaga de verdade com as DUAS condições: modo = apagar E a frase de confirmação certa.
CONFIRMACAO_ESPERADA = f"APAGAR {CATALOG}"
APAGAR = MODO == "apagar" and CONFIRMACAO == CONFIRMACAO_ESPERADA

if MODO == "apagar" and not APAGAR:
    print(f"⚠️  Modo 'apagar' escolhido, mas a confirmação não é '{CONFIRMACAO_ESPERADA}'. "
          "Fica em SIMULAÇÃO: nada será apagado.")
print("Modo:", "APAGAR (a sério)" if APAGAR else "SIMULAÇÃO (nada é apagado)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. O que o projeto criou
# MAGIC
# MAGIC Estes nomes foram copiados do notebook `movies_medallion_project_v3` (secção 0, Setup) e do ficheiro `job_filmes_medallion.yml`. Só objetos com estes nomes são apagados.

# COMMAND ----------

# Tabelas de cada camada, tal como estão no notebook do projeto
BRONZE_TABLES = ["tmdb_movie_lists", "tmdb_genres", "tmdb_movie_details",
                 "imdb_title_basics", "imdb_title_ratings",
                 "ml_ratings", "ml_movies", "ml_links", "ml_tags"]
SILVER_TABLES = ["tmdb_movies", "tmdb_movie_details", "tmdb_movie_genres",
                 "imdb_movies", "imdb_movie_genres",
                 "ml_ratings", "ml_movies", "ml_links", "ml_tags",
                 "movies_unified"]
# Star schema primeiro e com a tabela de factos à cabeça: tem chaves estrangeiras para as dimensões,
# por isso tem de ser apagada antes delas.
STAR_TABLES = ["fact_rating", "bridge_movie_genre", "dim_movie", "dim_source", "dim_year",
               "dim_language", "dim_genre"]
GOLD_TABLES = STAR_TABLES + [
    "ratings_by_decade", "gap_by_genre", "gap_by_language", "gap_by_budget",
    "movie_rating_gaps", "roi_by_genre", "tags_by_gap", "gap_model_coefficients", "dq_results",
    "genre_catalog", "user_activity", "top_movies", "ratings_over_time", "flag_sensitivity",
    "movie_rating_shifts", "movie_day_anomalies", "anomaly_participants"]

TABELAS_PROJETO = {"gold": GOLD_TABLES, "silver": SILVER_TABLES, "bronze": BRONZE_TABLES}

# O schema "backup" só existe se alguma vez correste o projeto com reiniciar = tudo_com_backup.
# As cópias chamam-se <camada>_<tabela>_<AAAAMMDD_HHMMSS>, por ex. bronze_ml_ratings_20261108_192200.
_todas = sorted(set(BRONZE_TABLES + SILVER_TABLES + GOLD_TABLES), key=len, reverse=True)
PADRAO_BACKUP = re.compile(r"^(bronze|silver|gold)_(" + "|".join(_todas) + r")_\d{8}_\d{6}$")

# Ordem de limpeza dos schemas (bronze por último, porque tem o Volume com os dados)
SCHEMAS_PROJETO = ["gold", "silver", "backup", "bronze"]
VOLUME_RAW = "raw"                                   # /Volumes/<catálogo>/bronze/raw
SCHEMAS_DO_SISTEMA = {"default", "information_schema"}  # criados automaticamente com o catálogo

# Ficheiros do projeto que podem estar no Workspace (os notebooks aparecem sem extensão)
NOMES_WORKSPACE = {"movies_medallion_project_v3", "movies_medallion_project_v3.py",
                   "movies_medallion_project_v3.ipynb",
                   "movies_medallion_project_v3_antes_legibilidade",
                   "movies_medallion_project_v3_antes_legibilidade.py",
                   "job_filmes_medallion.yml", "arquitetura_preview.png", "star_schema_preview.png"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Funções auxiliares
# MAGIC
# MAGIC A ideia é simples: cada função de **inventário** olha para o que existe e acrescenta linhas a um **plano** (uma lista). Cada linha diz o objeto, se é para `APAGAR` ou `MANTER`, porquê, e (se for para apagar) a função que o apaga. Primeiro mostramos o plano; só no fim, e só com confirmação, o executamos.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()                      # no Databricks autentica-se sozinho
EU = w.current_user.me().user_name         # o teu email
HOME = f"/Users/{EU}"                      # a tua pasta pessoal no Workspace

try:
    ESTE_NOTEBOOK = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
except Exception:
    ESTE_NOTEBOOK = None


def q(nome):
    """Põe um nome entre acentos graves para o SQL (ex.: `abd_project`)."""
    return f"`{nome}`"


def catalogo_existe():
    return CATALOG in [r[0] for r in spark.sql("SHOW CATALOGS").collect()]


def schemas_do_catalogo():
    return [r[0] for r in spark.sql(f"SHOW SCHEMAS IN {q(CATALOG)}").collect()]


def tabelas_do_schema(schema):
    linhas = spark.sql(f"SHOW TABLES IN {q(CATALOG)}.{q(schema)}").collect()
    return [r["tableName"] for r in linhas if not r["isTemporary"]]


def volumes_do_schema(schema):
    linhas = spark.sql(f"SHOW VOLUMES IN {q(CATALOG)}.{q(schema)}").collect()
    return [r.asDict().get("volume_name", r[1]) for r in linhas]


def caminho_api(p):
    """O Workspace mostra '/Workspace/Users/...', mas a API usa '/Users/...'."""
    p = p.strip().rstrip("/")
    return p[len("/Workspace"):] if p.startswith("/Workspace/") else p


def novo_plano():
    return []


def adicionar(plano, tipo, objeto, acao, motivo, executar=None):
    plano.append({"tipo": tipo, "objeto": objeto, "acao": acao, "motivo": motivo, "_executar": executar})


def mostrar(plano, titulo):
    df = pd.DataFrame([{k: v for k, v in p.items() if k != "_executar"} for p in plano],
                      columns=["tipo", "objeto", "acao", "motivo"])
    df.insert(0, "ordem", range(1, len(df) + 1))
    print(titulo)
    print(f"  APAGAR: {(df.acao == 'APAGAR').sum()}   |   MANTER: {(df.acao == 'MANTER').sum()}")
    if len(df):
        display(df)
    else:
        print("  (nada encontrado)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inventário: o que existe e o que acontece a cada objeto

# COMMAND ----------

# ------------------------------- 4.1 Job ---------------------------------------
def inventario_job(plano):
    """Procura Jobs com o nome exato do widget e guarda os notebooks que eles correm."""
    notebooks_do_job = set()
    for j in w.jobs.list(name=NOME_JOB):
        job = w.jobs.get(j.job_id)
        if job.settings is None or job.settings.name != NOME_JOB:
            continue
        for t in job.settings.tasks or []:
            if t.notebook_task and t.notebook_task.notebook_path:
                notebooks_do_job.add(caminho_api(t.notebook_task.notebook_path))
        nome = f"{NOME_JOB} (id {job.job_id})"
        if job.creator_user_name != EU:
            adicionar(plano, "Job", nome, "MANTER", f"criado por {job.creator_user_name}, não por ti")
        else:
            adicionar(plano, "Job", nome, "APAGAR", "Job do pipeline (apaga também o histórico de execuções)",
                      lambda job_id=job.job_id: w.jobs.delete(job_id))
    return notebooks_do_job


# ------------------------- 4.2 Unity Catalog e dados ---------------------------
def pertence_ao_projeto(schema, tabela):
    if schema == "backup":
        return bool(PADRAO_BACKUP.match(tabela))
    return tabela in TABELAS_PROJETO.get(schema, [])


def ordem_da_tabela(schema, tabela):
    """Ordem pela lista do projeto (factos antes das dimensões); as restantes vão no fim."""
    lista = TABELAS_PROJETO.get(schema, [])
    return lista.index(tabela) if tabela in lista else len(lista)


def inventario_catalogo(plano):
    if not catalogo_existe():
        adicionar(plano, "Catálogo", CATALOG, "MANTER", "não existe (nada a fazer)")
        return

    schemas = schemas_do_catalogo()
    schemas_mantidos = []

    for schema in SCHEMAS_PROJETO:
        if schema not in schemas:
            continue
        full_schema = f"{CATALOG}.{schema}"
        algo_fica = False

        for t in sorted(tabelas_do_schema(schema), key=lambda t: ordem_da_tabela(schema, t)):
            full = f"{q(CATALOG)}.{q(schema)}.{q(t)}"
            if pertence_ao_projeto(schema, t):
                adicionar(plano, "Tabela", f"{full_schema}.{t}", "APAGAR", "criada pelo projeto",
                          lambda full=full: spark.sql(f"DROP TABLE IF EXISTS {full}"))
            else:
                algo_fica = True
                adicionar(plano, "Tabela", f"{full_schema}.{t}", "MANTER", "não foi criada pelo projeto")

        for v in volumes_do_schema(schema):
            full = f"{q(CATALOG)}.{q(schema)}.{q(v)}"
            if schema == "bronze" and v == VOLUME_RAW and APAGAR_DADOS_RAW:
                adicionar(plano, "Volume (dados)", f"/Volumes/{CATALOG}/{schema}/{v}", "APAGAR",
                          "ficheiros IMDb, TMDB e MovieLens do projeto (apagados para sempre)",
                          lambda full=full: spark.sql(f"DROP VOLUME IF EXISTS {full}"))
            else:
                algo_fica = True
                motivo = ("widget 4 = nao (manter os dados)" if v == VOLUME_RAW and schema == "bronze"
                          else "não foi criado pelo projeto")
                adicionar(plano, "Volume (dados)", f"/Volumes/{CATALOG}/{schema}/{v}", "MANTER", motivo)

        if algo_fica:
            schemas_mantidos.append(schema)
            adicionar(plano, "Schema", full_schema, "MANTER", "ainda tem objetos que ficam (ver acima)")
        else:
            # Sem CASCADE: se por acaso ainda houver algo lá dentro, o Databricks recusa e nada se perde.
            adicionar(plano, "Schema", full_schema, "APAGAR", "schema do projeto (só se ficar vazio)",
                      lambda s=schema: spark.sql(f"DROP SCHEMA IF EXISTS {q(CATALOG)}.{q(s)}"))

    # ---- o catálogo em si ----
    outros = [s for s in schemas if s not in SCHEMAS_PROJETO and s not in SCHEMAS_DO_SISTEMA]
    default_tem_coisas = ("default" in schemas
                          and (tabelas_do_schema("default") or volumes_do_schema("default")))
    if not APAGAR_CATALOGO:
        adicionar(plano, "Catálogo", CATALOG, "MANTER", "widget 5 = nao")
    elif CATALOG in CATALOGOS_PROTEGIDOS:
        adicionar(plano, "Catálogo", CATALOG, "MANTER", "catálogo do sistema/partilhado: nunca é apagado")
    elif outros:
        adicionar(plano, "Catálogo", CATALOG, "MANTER", f"tem outros schemas que não são do projeto: {outros}")
    elif default_tem_coisas:
        adicionar(plano, "Catálogo", CATALOG, "MANTER", "o schema 'default' tem tabelas ou volumes")
    elif schemas_mantidos:
        adicionar(plano, "Catálogo", CATALOG, "MANTER", f"schemas mantidos: {schemas_mantidos}")
    else:
        adicionar(plano, "Catálogo", CATALOG, "APAGAR",
                  "só restam os schemas do projeto e o 'default' vazio", apagar_catalogo)


def apagar_catalogo():
    """Volta a confirmar, no momento de apagar, que o catálogo só tem 'default' vazio."""
    restantes = [s for s in schemas_do_catalogo() if s not in SCHEMAS_DO_SISTEMA]
    if restantes:
        raise RuntimeError(f"O catálogo ainda tem schemas {restantes}: não foi apagado.")
    if "default" in schemas_do_catalogo() and (tabelas_do_schema("default") or volumes_do_schema("default")):
        raise RuntimeError("O schema 'default' não está vazio: o catálogo não foi apagado.")
    # CASCADE aqui só leva 'default' (vazio) e 'information_schema', que o Databricks cria sozinho.
    spark.sql(f"DROP CATALOG IF EXISTS {q(CATALOG)} CASCADE")


# ------------------------------ 4.3 Workspace ----------------------------------
def motivo_para_manter(caminho, tipo):
    """Devolve a razão para NÃO apagar este caminho, ou None se for seguro apagá-lo."""
    if not caminho.startswith(HOME + "/"):
        return "fora da tua pasta pessoal"
    if ESTE_NOTEBOOK and (ESTE_NOTEBOOK == caminho or ESTE_NOTEBOOK.startswith(caminho + "/")):
        return "é (ou contém) este notebook de limpeza: apaga-o à mão no fim"
    if tipo == "DIRECTORY" and ESTE_NOTEBOOK is None:
        return "não consegui saber onde está este notebook; por segurança a pasta fica"
    return None


def inventario_workspace(plano, notebooks_do_job):
    candidatos = set(notebooks_do_job) | {caminho_api(c) for c in CAMINHOS_EXTRA}
    # procura na tua pasta pessoal os ficheiros com os nomes do projeto
    try:
        for obj in w.workspace.list(HOME, recursive=True):
            if obj.path.rsplit("/", 1)[-1] in NOMES_WORKSPACE:
                candidatos.add(obj.path)
    except Exception as e:
        print(f"Aviso: não consegui percorrer {HOME}: {e}")

    pastas_a_apagar = []
    for caminho in sorted(candidatos):   # ordem alfabética: uma pasta aparece antes do que tem dentro
        # se a pasta-mãe já vai ser apagada, este ficheiro vai com ela: não precisa de linha própria
        if any(caminho.startswith(p + "/") for p in pastas_a_apagar):
            continue
        try:
            info = w.workspace.get_status(caminho)
        except Exception:
            continue   # não existe: nada a fazer
        tipo = info.object_type.value if info.object_type else "?"
        motivo = motivo_para_manter(caminho, tipo)
        if motivo:
            adicionar(plano, "Workspace", f"{caminho} ({tipo})", "MANTER", motivo)
        else:
            descr = "pasta indicada no widget 7 (e tudo o que tem dentro)" if tipo == "DIRECTORY" \
                    else "ficheiro do projeto"
            if tipo == "DIRECTORY":
                pastas_a_apagar.append(caminho)
            adicionar(plano, "Workspace", f"{caminho} ({tipo})", "APAGAR", descr,
                      lambda c=caminho, rec=(tipo == "DIRECTORY"): w.workspace.delete(c, recursive=rec))


def construir_plano():
    plano = novo_plano()
    notebooks_do_job = inventario_job(plano)      # 1.º o Job
    inventario_catalogo(plano)                    # 2.º tabelas, dados, schemas, catálogo
    inventario_workspace(plano, notebooks_do_job) # 3.º ficheiros no Workspace
    return plano

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Plano (o que vai ser apagado)
# MAGIC
# MAGIC Lê esta tabela antes de passar ao modo `apagar`. A coluna `ordem` é a ordem em que as coisas são apagadas.

# COMMAND ----------

PLANO = construir_plano()
mostrar(PLANO, f"Plano de limpeza do catálogo '{CATALOG}', do Job '{NOME_JOB}' e do Workspace de {EU}")
if not APAGAR:
    print(f"\nSIMULAÇÃO: nada foi apagado. Para apagar: widget 2 = apagar, "
          f"widget 3 = '{CONFIRMACAO_ESPERADA}', e Run all.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Executar (só com confirmação)

# COMMAND ----------

if APAGAR:
    resultados = []
    for passo in PLANO:
        if passo["acao"] != "APAGAR":
            continue
        try:
            passo["_executar"]()
            estado = "apagado"
        except Exception as e:
            estado = f"ERRO: {str(e).splitlines()[0][:200]}"
        resultados.append({"tipo": passo["tipo"], "objeto": passo["objeto"], "resultado": estado})
        print(f"[{estado[:7]}] {passo['tipo']}: {passo['objeto']}")
    display(pd.DataFrame(resultados))
else:
    print("Simulação: este passo foi saltado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verificação final
# MAGIC
# MAGIC Volta a fazer o inventário. Depois de uma limpeza bem feita, não deve sobrar nenhuma linha `APAGAR`.

# COMMAND ----------

if APAGAR:
    falta = [p for p in construir_plano() if p["acao"] == "APAGAR"]
    if falta:
        mostrar(falta, "⚠️  Ainda há objetos por apagar (ver erros no passo 6 e voltar a correr):")
    else:
        print("✅ Limpeza completa: já não há nada do projeto para apagar.")
        if ESTE_NOTEBOOK:
            print(f"Falta só apagar este notebook à mão, se quiseres: {ESTE_NOTEBOOK}")
    # Remove o widget da confirmação para não repetir por engano (volta vazio no próximo Run all)
    try:
        dbutils.widgets.remove("confirmacao")
    except Exception:
        pass
else:
    print("Simulação: nada a verificar.")
