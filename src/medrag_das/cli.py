"""`medrag` command-line interface.

    medrag build-kb [--limit N]
    medrag train [--all | --category SLUG]
    medrag eval [--methods ...] [--datasets ...] [--limit N]
    medrag ask --question ... --options "a|b|c|d" [--category SLUG]

Commands wire config + backend + the unit modules together. Where an underlying unit is a
stub, the command prints a clear message pointing at the module to implement.
"""
from __future__ import annotations

import typer

from .config import load_config

app = typer.Typer(add_completion=False, help="DAS + RAG parroting-reduction system (reconstruction).")

_BANNER = (
    "⚠️  Research proof-of-concept — NOT a medical device. Outputs must not inform clinical "
    "decisions. See WARNINGS.md."
)


@app.callback()
def _main() -> None:
    typer.secho(_BANNER, fg=typer.colors.YELLOW)


@app.command("build-kb")
def build_kb(limit: int = typer.Option(None, help="Max abstracts (default: config kb_limit).")):
    """Embed PubMed abstracts into a FAISS IndexFlatL2 (retrieval/build_kb.py)."""
    cfg = load_config()
    from .retrieval import build_kb as kb

    n = limit or cfg.retrieval.kb_limit
    typer.echo(f"Building KB: {n} abstracts -> {cfg.retrieval.kb_path}")
    kb.build(cfg.retrieval.kb_path, cfg.retrieval.meta_path, limit=n)
    typer.secho("KB build complete.", fg=typer.colors.GREEN)


@app.command("train")
def train(
    all_: bool = typer.Option(False, "--all", help="Train all 21 adapters."),
    category: str = typer.Option(None, help="Train a single category slug."),
):
    """Train QLoRA adapters (adapters/train_all.py)."""
    cfg = load_config()
    if not all_ and not category:
        raise typer.BadParameter("pass --all or --category SLUG")

    from .adapters.train_all import train_all
    from .backends import get_backend

    backend = get_backend(cfg.backend)
    backend.load_base(cfg.models["base_model"], cfg.models["quantize_bits"])

    typer.echo("Loading MedMCQA train split...")
    from datasets import load_dataset  # noqa: F401 -- imported lazily

    split = load_dataset(cfg.adapters.dataset, split="train")
    only = None if all_ else [category]
    results = train_all(backend, cfg, split, only=only)
    for slug, m in results.items():
        typer.echo(f"  {slug}: {m}")
    typer.secho("Training complete.", fg=typer.colors.GREEN)


@app.command("eval")
def eval_cmd(
    methods: list[str] = typer.Option(list(("baseline", "das", "rag", "das_rag")), help="Methods."),
    datasets: list[str] = typer.Option(list(("medmcqa", "nota", "fake")), help="Datasets."),
    limit: int = typer.Option(None, help="Cap questions per dataset (debug)."),
):
    """Run the 4-method × 3-dataset evaluation matrix (evaluation/run_eval.py)."""
    from .evaluation.run_eval import run_matrix

    cfg = load_config()
    run_matrix(cfg, methods=methods, datasets=datasets, limit=limit)


@app.command("ask")
def ask(
    question: str = typer.Option(..., help="The question stem."),
    options: str = typer.Option(..., help="Answer choices separated by '|'."),
    category: str = typer.Option(None, help="Oracle category slug (for DAS)."),
    method: str = typer.Option("das_rag", help="baseline|das|rag|das_rag"),
):
    """Ask one question through the full pipeline (loads only what `method` needs)."""
    cfg = load_config()
    opts = [o.strip() for o in options.split("|") if o.strip()]

    from .adapters.registry import AdapterRegistry
    from .adapters.selector import build_selector
    from .backends import get_backend
    from .config import REPO_ROOT
    from .evaluation.run_eval import build_retriever
    from .pipeline import Pipeline

    backend = get_backend(cfg.backend)
    backend.load_base(cfg.models["base_model"], cfg.models["quantize_bits"])

    registry = selector = retriever = None
    if method in ("das", "das_rag"):
        registry = AdapterRegistry(backend, REPO_ROOT / cfg.adapters.dir, cfg.category_slugs)
        selector = build_selector(
            cfg.selection["strategy"], cfg.category_slugs,
            fake_strategy=cfg.selection["fake_strategy"], backend=backend,
        )
    if method in ("rag", "das_rag"):
        retriever = build_retriever(cfg)

    pipe = Pipeline(backend, registry=registry, selector=selector, retriever=retriever,
                    gen_cfg=cfg.generation)
    out = pipe.run(method, question, opts, gold_category_slug=category)

    typer.echo(f"\nadapter: {out.adapter_slug}   retrieved: {len(out.context)} abstract(s)")
    typer.echo(f"parsed  : cop_index={out.answer.cop_index} abstained={out.answer.abstained} "
               f"parse_failed={out.answer.parse_failed}")
    typer.secho("\n--- raw generation ---", fg=typer.colors.CYAN)
    typer.echo(out.answer.raw)


if __name__ == "__main__":
    app()
