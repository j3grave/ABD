"""
Descarrega dados do TMDB para ficheiros JSON Lines, para carregar num Volume do Databricks.

Mostra o estado dos ficheiros locais antes de começar, barras de progresso durante o
download e, opcionalmente, o upload para o Volume.

Uso (no teu computador):
    pip install requests
    python tmdb_download.py                      # pergunta o que fazer se já existirem ficheiros
    python tmdb_download.py --reuse              # reutiliza os ficheiros existentes (só descarrega os que faltam)
    python tmdb_download.py --force              # volta a descarregar tudo
    python tmdb_download.py --ids-file movielens_data/links.csv   # + detalhe dos filmes do MovieLens (recomendado)
    python tmdb_download.py --upload /Volumes/abd_project/bronze/raw/tmdb   # faz também o upload (Databricks CLI)

O token (Read Access Token v4 do TMDB) é lido da variável de ambiente TMDB_TOKEN;
se não existir, o script pede-o no terminal (só se houver algo para descarregar).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from getpass import getpass
from pathlib import Path

import requests

# ---------------- Configuração ----------------
LISTS = ["popular", "top_rated", "now_playing"]
PAGES = 5              # 20 filmes por página
LANGUAGE = "pt-PT"
DETAILS_LIMIT = 200    # nº de filmes (mais populares) para os quais se pede o detalhe
DEFAULT_OUT_DIR = "tmdb_data"
BASE_URL = "https://api.themoviedb.org/3"
FILES = {
    "lists": "movie_lists.jsonl",
    "genres": "genres.jsonl",
    "details": "movie_details.jsonl",
}
# ----------------------------------------------


# ============ Barra de progresso ============
class ProgressBar:
    """Barra de progresso simples (sem dependências), segura para uso com threads."""

    def __init__(self, total, label, width=30):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.done = 0
        self.lock = threading.Lock()
        self.start = time.time()
        self._render()

    def update(self, n=1):
        with self.lock:
            self.done = min(self.done + n, self.total)
            self._render()

    def _render(self):
        frac = self.done / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0
        sys.stdout.write(
            f"\r{self.label:<24}[{bar}] {frac * 100:5.1f}% ({self.done}/{self.total}) "
            f"{elapsed:4.0f}s  resta ~{eta:3.0f}s "
        )
        sys.stdout.flush()

    def close(self, note=""):
        with self.lock:
            self.done = self.total
            self._render()
            sys.stdout.write(f"\n  {note}\n" if note else "\n")
            sys.stdout.flush()


# ============ Estado dos ficheiros ============
def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def describe(path):
    """Devolve informação sobre o ficheiro, ou None se não existir."""
    if not path.exists():
        return None
    st = path.stat()
    with open(path, encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    return {"size": st.st_size, "lines": lines,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")}


def print_status(title, paths):
    print(f"\n{title}:")
    for key, path in paths.items():
        info = describe(path)
        if info is None:
            print(f"  [ ] {path.name:<22} não existe")
        elif info["lines"] == 0:
            print(f"  [!] {path.name:<22} existe mas está vazio")
        else:
            print(f"  [x] {path.name:<22} {human_size(info['size']):>9}  "
                  f"{info['lines']:>6} linhas  (modificado {info['mtime']})")


def is_valid(path):
    info = describe(path)
    return info is not None and info["lines"] > 0


def ensure_dir(directory: Path) -> None:
    """Verifica se a diretoria existe; se não existir, cria-a (incluindo as pastas-pai)."""
    if directory.is_dir():
        print(f"Diretoria já existe: {directory.resolve()}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Diretoria criada: {directory.resolve()}")


def decide_downloads(paths, args):
    """Decide que ficheiros descarregar, tendo em conta os que já existem."""
    existing = [k for k, p in paths.items() if is_valid(p)]
    todo = {k: True for k in paths}

    if existing and not args.force:
        reuse = args.reuse
        if not args.reuse:
            if sys.stdin.isatty():
                answer = input("\nJá existem ficheiros. [R]eutilizar ou [S]obrescrever? (R/s): ").strip().lower()
                reuse = answer != "s"
            else:
                reuse = True  # sem terminal interativo: não apaga nada por omissão
        if reuse:
            for k in existing:
                todo[k] = False

    # O detalhe depende das listas: se as listas forem refeitas, refaz-se também o detalhe.
    if todo["lists"] and not todo["details"]:
        todo["details"] = True
        print("Nota: as listas vão ser refeitas, por isso o detalhe também.")
    return todo


# ============ API do TMDB ============
_session = None


def get_session():
    """Cria a sessão HTTP só quando é preciso (evita pedir o token se tudo for reutilizado)."""
    global _session
    if _session is None:
        token = os.environ.get("TMDB_TOKEN") or getpass("TMDB Read Access Token: ").strip()
        _session = requests.Session()
        _session.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json"})
    return _session


def tmdb_get(endpoint, params=None, retries=4):
    session = get_session()
    params = {"language": LANGUAGE, **(params or {})}
    for attempt in range(retries):
        r = session.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def write_jsonl(path, rows):
    """Escreve num ficheiro temporário e só no fim substitui o final (evita ficheiros a meio)."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ============ Passos ============
def step_lists(path):
    list_rows, popularity = [], {}
    bar = ProgressBar(len(LISTS) * PAGES, "Listas de filmes")
    for list_name in LISTS:
        for page in range(1, PAGES + 1):
            data = tmdb_get(f"/movie/{list_name}", {"page": page})
            for movie in data["results"]:
                list_rows.append({"list_name": list_name, "page": page,
                                  "payload": json.dumps(movie, ensure_ascii=False)})
                popularity[movie["id"]] = max(popularity.get(movie["id"], 0), movie.get("popularity", 0))
            bar.update()
            if page >= data.get("total_pages", page):
                bar.update(PAGES - page)  # páginas que já não existem
                break
    write_jsonl(path, list_rows)
    bar.close(f"{path.name}: {len(list_rows)} linhas")
    return popularity


def popularity_from_file(path):
    """Quando as listas são reutilizadas, recupera a popularidade a partir do ficheiro."""
    popularity = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            movie = json.loads(json.loads(line)["payload"])
            popularity[movie["id"]] = max(popularity.get(movie["id"], 0), movie.get("popularity", 0))
    return popularity


def step_genres(path):
    bar = ProgressBar(1, "Géneros")
    genres = tmdb_get("/genre/movie/list")["genres"]
    write_jsonl(path, [{"payload": json.dumps(g, ensure_ascii=False)} for g in genres])
    bar.close(f"{path.name}: {len(genres)} linhas")


def ids_from_links(path, limit):
    """Lê a coluna tmdbId do links.csv do MovieLens (ignora vazios) até ao limite indicado."""
    import csv
    ids = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = (row.get("tmdbId") or "").strip()
            if value.isdigit():
                ids.append(int(value))
            if len(ids) >= limit:
                break
    return ids


def step_details(path, popularity, extra_ids=()):
    top_ids = sorted(popularity, key=popularity.get, reverse=True)[:DETAILS_LIMIT]
    top_ids = list(dict.fromkeys(top_ids + list(extra_ids)))  # sem repetidos, mantém a ordem
    bar = ProgressBar(len(top_ids), "Detalhe dos filmes")
    failures = []

    def fetch_detail(movie_id):
        try:
            return {"movie_id": movie_id,
                    "payload": json.dumps(tmdb_get(f"/movie/{movie_id}"), ensure_ascii=False)}
        except Exception as e:
            failures.append((movie_id, str(e)))
            return None
        finally:
            bar.update()

    get_session()  # pede o token antes de lançar as threads
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(fetch_detail, top_ids) if r]
    write_jsonl(path, rows)
    bar.close(f"{path.name}: {len(rows)} linhas")
    for movie_id, err in failures:
        print(f"  Falhou {movie_id}: {err}")


def step_upload(paths, volume_path, profile=None):
    """Upload opcional para o Volume, com a Databricks CLI (databricks fs cp)."""
    cli = shutil.which("databricks")
    if not cli:
        print("\nUpload ignorado: a Databricks CLI não está instalada ou não está no PATH.")
        print(f"Carrega os ficheiros manualmente para {volume_path} (Catalog > Upload to this volume).")
        return
    bar = ProgressBar(len(paths), "Upload para o Volume")
    for path in paths:
        dest = f"dbfs:{volume_path.rstrip('/')}/{path.name}"
        cmd = [cli, "fs", "cp", str(path), dest, "--overwrite"]
        if profile:
            cmd += ["--profile", profile]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stdout.write("\n")
            print(f"Falhou o upload de {path.name}:\n{result.stderr.strip()}")
            sys.exit(1)
        bar.update()
    bar.close(f"{len(paths)} ficheiros enviados para {volume_path}")


# ============ Programa principal ============
def parse_args():
    p = argparse.ArgumentParser(description="Descarrega dados do TMDB para JSON Lines.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="diretoria local de destino")
    p.add_argument("--force", action="store_true", help="volta a descarregar tudo")
    p.add_argument("--reuse", action="store_true", help="reutiliza os ficheiros existentes")
    p.add_argument("--upload", metavar="VOLUME_PATH",
                   help="faz upload para o Volume, ex.: /Volumes/abd_project/bronze/raw/tmdb")
    p.add_argument("--profile", help="profile da Databricks CLI (para o upload)")
    p.add_argument("--ids-file", metavar="LINKS_CSV",
                   help="links.csv do MovieLens: pede também o detalhe destes filmes (coluna tmdbId)")
    p.add_argument("--ids-limit", type=int, default=1000,
                   help="máximo de filmes lidos do --ids-file (por defeito 1000)")
    # Dentro do Databricks, o sys.argv é do launcher do kernel (-f connection.json...),
    # o que faria o argparse sair com SystemExit(2). Este script é para o terminal local.
    if not (sys.argv and sys.argv[0].endswith("tmdb_download.py")):
        raise RuntimeError(
            "Este script é pensado para correr no teu computador: python tmdb_download.py. "
            "No workspace, o argparse recebe os argumentos do kernel (SystemExit: 2) e o "
            "compute serverless não tem acesso à API do TMDB. Corre-o localmente e carrega "
            "os .jsonl para o Volume (ou usa --upload).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    paths = {key: out_dir / name for key, name in FILES.items()}

    print_status("Estado dos ficheiros locais (antes)", paths)
    todo = decide_downloads(paths, args)

    for key, path in paths.items():
        if not todo[key]:
            print(f"\nJá existe, download ignorado: {path.name}")

    popularity = None
    if todo["lists"]:
        print()
        popularity = step_lists(paths["lists"])
    if todo["genres"]:
        print()
        step_genres(paths["genres"])
    if todo["details"]:
        print()
        if popularity is None:
            popularity = popularity_from_file(paths["lists"])
        extra_ids = ids_from_links(args.ids_file, args.ids_limit) if args.ids_file else []
        if args.ids_file:
            print(f"{len(extra_ids)} filmes lidos de {args.ids_file}")
        step_details(paths["details"], popularity, extra_ids)

    print_status("Estado dos ficheiros locais (depois)", paths)

    if args.upload:
        print()
        step_upload(list(paths.values()), args.upload, args.profile)
    else:
        print(f"\nConcluído. Carrega os 3 ficheiros de '{out_dir}' para o Volume do Databricks "
              f"(ou usa --upload /Volumes/<catalogo>/<schema>/<volume>).")


if __name__ == "__main__":
    main()
