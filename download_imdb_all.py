"""
Descarrega TODOS os ficheiros dos datasets do IMDb (os 7 .tsv.gz) para uma pasta local.

Ao contrário do download_imdb_movielens.py, não filtra nada: guarda os ficheiros tal como o IMDb os publica.
Reutiliza a barra de progresso desse script (têm de estar na mesma pasta).

Uso:
    pip install requests
    python download_imdb_all.py                              # os 7 ficheiros para imdb_data_full\
    python download_imdb_all.py --only title.basics.tsv.gz title.ratings.tsv.gz
    python download_imdb_all.py --force                      # volta a descarregar
    python download_imdb_all.py --upload /Volumes/abd_project/landing/raw   # + upload (Databricks CLI)

Datasets do IMDb: apenas para uso pessoal e não comercial (https://developer.imdb.com/non-commercial-datasets/).
Alguns ficheiros (principals, akas, name.basics) são grandes: confirma o espaço em disco.
"""
import argparse
import sys
from pathlib import Path

from download_imdb_movielens import IMDB_BASE, download, ensure_dir, print_status
from tmdb_download import step_upload

IMDB_FILES = [
    "name.basics.tsv.gz",
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz",
]


def parse_args():
    p = argparse.ArgumentParser(description="Descarrega todos os ficheiros dos datasets do IMDb.")
    p.add_argument("--out-dir", default="imdb_data_full", help="pasta local de destino")
    p.add_argument("--only", nargs="+", choices=IMDB_FILES, metavar="FICHEIRO",
                   help="descarrega só estes ficheiros (por defeito, todos)")
    p.add_argument("--force", action="store_true", help="volta a descarregar mesmo que já existam")
    p.add_argument("--base-url", default=IMDB_BASE, help=argparse.SUPPRESS)
    p.add_argument("--upload", metavar="RAW_ROOT",
                   help="raiz do Volume, ex.: /Volumes/abd_project/landing/raw (cria a subpasta imdb/)")
    p.add_argument("--profile", help="profile da Databricks CLI (para o upload)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    names = args.only or IMDB_FILES
    paths = [out_dir / n for n in names]

    print_status("Ficheiros locais (antes)", paths)
    failed = []
    for name, path in zip(names, paths):
        if path.exists() and not args.force:
            print(f"\nJá existe, download ignorado: {name}")
            continue
        print()
        try:
            download(f"{args.base_url}/{name}", path, f"Download {name[:14]}")
        except Exception as e:  # um ficheiro a falhar não impede os outros
            failed.append(name)
            print(f"Falhou {name}: {e}")

    print_status("Ficheiros locais (depois)", paths)
    if failed:
        print(f"\nFalharam {len(failed)} ficheiro(s): {failed}. Volta a correr o script (os que existem são ignorados).")
        sys.exit(1)

    if args.upload:
        print()
        step_upload([p for p in paths if p.exists()], f"{args.upload.rstrip('/')}/imdb", args.profile)
    else:
        print("\nConcluído. Carrega os ficheiros para o Volume (landing/raw/imdb), ou usa --upload.")


if __name__ == "__main__":
    main()
