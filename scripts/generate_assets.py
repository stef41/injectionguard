"""Generate SVG assets for injectionguard README."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def generate_detection_report():
    console = Console(record=True, width=90)

    table = Table(title="injectionguard scan results", show_header=True, header_style="bold red")
    table.add_column("Threat", style="bold")
    table.add_column("Strategy", style="cyan")
    table.add_column("Detection")

    table.add_row("[red]CRITICAL[/]", "heuristic", "Direct instruction override attempt")
    table.add_row("[red]CRITICAL[/]", "structural", "ChatML system injection (<|im_start|>system)")
    table.add_row("[red]CRITICAL[/]", "heuristic", "Jailbreak attempt (DAN mode)")
    table.add_row("[red]CRITICAL[/]", "structural", "End-of-text token injection")
    table.add_row("[yellow]HIGH[/]", "heuristic", "Role reassignment (you are now...)")
    table.add_row("[yellow]HIGH[/]", "encoding", "Base64 encoded injection detected")
    table.add_row("[yellow]HIGH[/]", "heuristic", "System prompt extraction attempt")
    table.add_row("[blue]MEDIUM[/]", "encoding", "Invisible Unicode character U+200B")
    table.add_row("[blue]MEDIUM[/]", "structural", "Code block contains injection content")
    table.add_row("[dim]LOW[/]", "structural", "Excessive newlines (context pushing)")

    console.print(table)
    console.print("\n[bold red]⚠ 10 injection patterns detected[/] • Threat level: [bold red]CRITICAL[/]")
    return console.export_svg(title="injectionguard detections")


def generate_strategies_overview():
    console = Console(record=True, width=85)

    table = Table(title="detection strategies", show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="bold")
    table.add_column("Patterns", justify="right", style="yellow")
    table.add_column("Detects")

    table.add_row("Heuristic", "30+", "Instruction override, jailbreaks, role manipulation")
    table.add_row("Encoding", "4", "Base64, hex, URL-encoded, Unicode obfuscation")
    table.add_row("Structural", "16+", "ChatML/Llama tokens, delimiters, padding")
    table.add_row("MCP", "3", "Role tags, instruction tags, conversation markers")

    console.print(table)
    console.print("\n[bold green]50+ detection patterns[/] • Zero dependencies • <1ms per scan")
    return console.export_svg(title="injectionguard strategies")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    svg = generate_detection_report()
    with open("assets/detection_report.svg", "w") as f:
        f.write(svg)
    print(f"  detection_report.svg: {len(svg):,} bytes")

    svg = generate_strategies_overview()
    with open("assets/strategies_overview.svg", "w") as f:
        f.write(svg)
    print(f"  strategies_overview.svg: {len(svg):,} bytes")
