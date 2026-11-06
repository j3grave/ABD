"""
Descarrega o IMDb (só filmes) e o MovieLens para pastas locais, prontas a carregar no Volume.

Reutiliza a barra de progresso e a lógica de upload do tmdb_download.py (têm de estar na mesma pasta).

Uso:
    pip install requests
    python download_imdb_movielens.py                       # IMDb e MovieLens
    python download_imdb_movielens.py --sources imdb        # só o IMDb
    python download_imdb_movielens.py --force               # refaz tudo
    python download_imdb_movielens.py --upload /Volumes/abd_project/landing/raw   # + upload (Databricks CLI)

Resultado:
    imdb_data/title.basics.movies.tsv.gz     (só titleType == movie)
    imdb_data/title.ratings.movies.tsv.gz    (só filmes)
    movielens_data/{ratings,movies,links,tags}.csv

Os ficheiros originais ficam em <pasta>/_raw (cache); não é preciso carregá-los para o Databricks.
Datasets do IMDb: apenas para uso pessoal e não comercial (https://developer.imdb.com/non-commercial-datasets/).
"""
import argparse
import gzip
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from tmdb_download import ProgressBar, ensure_dir, human_size, step_upload

IMDB_BASE = "https://datasets.imdbws.com"
ML_BASE = "https://files.grouplens.org/datasets/movielens"
IMDB_RAW = {"basics": "title.basics.tsv.gz", "ratings": "title.ratings.tsv.gz"}
IMDB_OUT = {"basics": "title.basics.movies.tsv.gz", "ratings": "title.ratings.movies.tsv.gz"}
ML_FILES = ["ratings.csv", "movies.csv", "links.csv", "tags.csv"]
HEADERS = {"User-Agent": "Mozilla/5.0 (projeto-universitario)"}


class ByteProgress(ProgressBar):
    """Barra de progresso em MB (para downloads e leitura de ficheiros grandes)."""

    def _render(self):
        frac = self.done / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        sys.stdout.write(
            f"\r{self.label:<24}[{bar}] {frac * 100:5.1f}% "
            f"({self.done / 1048576:7.1f}/{self.total / 1048576:7.1f} MB) {elapsed:4.0f}s "
        )
        sys.stdout.flush()

    def set(self, value):
        with self.lock:
            self.done = min(value, self.total)
            self._render()


# ============ Estado dos ficheiros ============
def count_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def print_status(title, paths):
    print(f"\n{title}:")
    for path in paths:
        if not path.exists():
            print(f"  [ ] {path.name:<30} não existe")
            continue
        st = path.stat()
        modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [x] {path.name:<30} {human_size(st.st_size):>9}  "
              f"{count_lines(path):>8} linhas  (modificado {modified})")


# ============ Download e filtragem ============
def download(url, dest, label):
    """Download com barra de progresso; escreve em .part e só no fim substitui o ficheiro final."""
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    received, declared = 0, 0
    try:
        with requests.get(url, stream=True, timeout=60, headers=HEADERS) as r:
            r.raise_for_status()
            declared = int(r.headers.get("Content-Length", 0))
            bar = ByteProgress(declared or 1, label)
            with open(tmp, "wb") as f:
                # r.raw: bytes tal como o servidor os envia (os .gz não podem ser alterados)
                for chunk in iter(lambda: r.raw.read(256 * 1024), b""):
                    f.write(chunk)
                    received += len(chunk)
                    bar.update(len(chunk))
        if declared and received != declared:  # ligação cortada a meio
            raise IOError(f"{received} de {declared} bytes")
    except Exception as e:  # nunca deixar um ficheiro truncado (nem o .part) para trás
        tmp.unlink(missing_ok=True)
        sys.stdout.write("\n")
        raise IOError(f"Download incompleto de {dest.name}: {e}") from e
    os.replace(tmp, dest)
    bar.close()


def filter_tsv(src, dest, keep_row, label):
    """Copia o cabeçalho e as linhas para as quais keep_row(parts) é verdadeira.
    Devolve o conjunto de tconst mantidos."""
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    bar = ByteProgress(src.stat().st_size, label)
    kept = set()
    with open(src, "rb") as raw, \
            gzip.open(raw, "rt", encoding="utf-8", newline="") as fin, \
            gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as fout:
        fout.write(fin.readline().rstrip("\r\n") + "\n")  # cabeçalho
        for i, line in enumerate(fin, 1):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if keep_row(parts):
                fout.write(line + "\n")
                kept.add(parts[0])
            if i % 50000 == 0:
                bar.set(raw.tell())
    os.replace(tmp, dest)
    bar.close(f"{dest.name}: {len(kept):,} linhas")
    return kept


def run_imdb(out_dir, force):
    raw_dir = out_dir / "_raw"
    outs = {k: out_dir / v for k, v in IMDB_OUT.items()}
    print_status("IMDb – ficheiros locais (antes)", list(outs.values()))

    if all(p.exists() for p in outs.values()) and not force:
        print("\nFicheiros do IMDb já existem, download ignorado (use --force para refazer).")
        return list(outs.values())

    for key, name in IMDB_RAW.items():
        raw = raw_dir / name
        if raw.exists() and not force:
            print(f"\nJá existe, download ignorado: {raw}")
        else:
            print()
            download(f"{IMDB_BASE}/{name}", raw, f"Download {name[:14]}")

    print()
    movies = filter_tsv(raw_dir / IMDB_RAW["basics"], outs["basics"],
                        lambda p: len(p) > 1 and p[1] == "movie", "Filtrar title.basics")
    print()
    filter_tsv(raw_dir / IMDB_RAW["ratings"], outs["ratings"],
               lambda p: p[0] in movies, "Filtrar title.ratings")

    print_status("IMDb – ficheiros locais (depois)", list(outs.values()))
    return list(outs.values())


def run_movielens(out_dir, version, force):
    outs = [out_dir / n for n in ML_FILES]
    print_status("MovieLens – ficheiros locais (antes)", outs)

    if all(p.exists() for p in outs) and not force:
        print("\nFicheiros do MovieLens já existem, download ignorado (use --force para refazer).")
        return outs

    zip_path = out_dir / "_raw" / f"{version}.zip"
    print()
    if zip_path.exists() and not force:
        print(f"Já existe, download ignorado: {zip_path}")
    else:
        download(f"{ML_BASE}/{version}.zip", zip_path, f"Download {version}")

    print()
    bar = ProgressBar(len(ML_FILES), "Extrair ficheiros")
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for name in ML_FILES:
            if name not in members:
                sys.stdout.write("\n")
                raise SystemExit(f"'{name}' não existe em {zip_path.name}")
            tmp = out_dir / (name + ".part")
            with z.open(members[name]) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, out_dir / name)
            bar.update()
    bar.close()

    print_status("MovieLens – ficheiros locais (depois)", outs)
    return outs


def parse_args():
    p = argparse.ArgumentParser(description="Descarrega IMDb (filmes) e MovieLens.")
    p.add_argument("--sources", choices=["imdb", "movielens", "both"], default="both")
    p.add_argument("--imdb-dir", default="imdb_data", help="pasta local do IMDb")
    p.add_argument("--movielens-dir", default="movielens_data", help="pasta local do MovieLens")
    p.add_argument("--movielens-version", default="ml-latest-small",
                   help="versão do MovieLens (ex.: ml-latest-small, ml-latest)")
    p.add_argument("--force", action="store_true", help="volta a descarregar e filtrar tudo")
    p.add_argument("--upload", metavar="RAW_ROOT",
                   help="raiz do Volume, ex.: /Volumes/abd_project/landing/raw (cria imdb/ e movielens/)")
    p.add_argument("--profile", help="profile da Databricks CLI (para o upload)")
    # Em notebook/IDE, sys.argv pertence ao kernel; usar defaults a menos que o
    # script seja mesmo corrido pela linha de comandos (python download_imdb_movielens.py ...)
    argv = None if Path(sys.argv[0]).name == "download_imdb_movielens.py" else []
    return p.parse_args(argv)


def main():
    args = parse_args()
    uploads = []  # (ficheiros, subpasta no Volume)

    if args.sources in ("imdb", "both"):
        uploads.append((run_imdb(Path(args.imdb_dir), args.force), "imdb"))
    if args.sources in ("movielens", "both"):
        uploads.append((run_movielens(Path(args.movielens_dir), args.movielens_version, args.force), "movielens"))

    if args.upload:
        for files, sub in uploads:
            print()
            step_upload(files, f"{args.upload.rstrip('/')}/{sub}", args.profile)
    else:
        print("\nConcluído. Carrega as pastas para o Volume (landing/raw/imdb e landing/raw/movielens), "
              "ou usa --upload /Volumes/<catalogo>/landing/raw")


if __name__ == "__main__":
    main()
