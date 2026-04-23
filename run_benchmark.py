from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Rich (cp1252 → utf-8)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from rich import print
from rich.progress import Progress, MofNCompleteColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/hotpot100_run",
    reflexion_attempts: int = 3,
    api_key: str = "",
) -> None:
    """
    Chạy benchmark ReAct và Reflexion Agent trên dataset.

    Tham số:
        dataset:            Đường dẫn đến file JSON dataset
        out_dir:            Thư mục output (sẽ được tạo tự động)
        reflexion_attempts: Số lần thử tối đa của ReflexionAgent
        api_key:            OpenAI API key (hoặc set env OPENAI_API_KEY)
    """
    # Set API key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.environ.get("OPENAI_API_KEY"):
        print("[red]ERROR: OPENAI_API_KEY not set. Use --api-key or set env var.[/red]")
        raise typer.Exit(1)

    examples = load_dataset(dataset)
    print(f"[bold]Loaded [cyan]{len(examples)}[/cyan] examples from [yellow]{dataset}[/yellow][/bold]")
    print(f"[bold]Expected num_records: [cyan]{len(examples) * 2}[/cyan] (ReAct + Reflexion)[/bold]\n")

    react     = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    react_records:     list = []
    reflexion_records: list = []

    # ── Chạy ReAct ───────────────────────────────────────────────────────
    print("[bold blue]--- Running ReAct Agent (1 attempt each) ---[/bold blue]")
    t_start = time.time()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("ReAct", total=len(examples))
        for ex in examples:
            record = react.run(ex)
            react_records.append(record)
            status = "OK" if record.is_correct else "XX"
            progress.update(task, advance=1, description=f"ReAct [{status}] {ex.qid}")

    react_acc = sum(1 for r in react_records if r.is_correct) / len(react_records)
    react_cost = sum(r.token_cost_usd for r in react_records)
    print(f"  ReAct accuracy: [green]{react_acc:.2%}[/green]  |  cost: [yellow]${react_cost:.6f}[/yellow]  |  time: {time.time()-t_start:.1f}s\n")

    # ── Chạy Reflexion ────────────────────────────────────────────────────
    print(f"[bold blue]--- Running Reflexion Agent (max {reflexion_attempts} attempts each) ---[/bold blue]")
    t_start = time.time()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Reflexion", total=len(examples))
        for ex in examples:
            record = reflexion.run(ex)
            reflexion_records.append(record)
            status = "OK" if record.is_correct else "XX"
            progress.update(
                task,
                advance=1,
                description=f"Reflexion [{status}] {ex.qid} (attempts={record.attempts})",
            )

    reflex_acc = sum(1 for r in reflexion_records if r.is_correct) / len(reflexion_records)
    reflex_cost = sum(r.token_cost_usd for r in reflexion_records)
    print(f"  Reflexion accuracy: [green]{reflex_acc:.2%}[/green]  |  cost: [yellow]${reflex_cost:.6f}[/yellow]  |  time: {time.time()-t_start:.1f}s\n")

    # ── Lưu kết quả ────────────────────────────────────────────────────
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)

    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report   = build_report(all_records, dataset_name=Path(dataset).name, mode="real")
    json_path, md_path = save_report(report, out_path)

    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print()
    print("[bold]== Summary ==[/bold]")
    print(json.dumps(report.summary, indent=2))
    print()
    print("[bold]== Cost Summary ==[/bold]")
    print(json.dumps(report.meta.get("cost_summary", {}), indent=2))


if __name__ == "__main__":
    app()
