import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.json import JSON
from rich.syntax import Syntax
from rich.spinner import Spinner

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

console = Console()

# ─── Prompt Toolkit Setup ────────────────────────────────────────────────────────
commands = ['exit', 'quit', 'help', 'list tools']
completer = WordCompleter(commands, ignore_case=True)
session = PromptSession(
    completer=completer,
    history=InMemoryHistory(),
    style=Style.from_dict({"": "#00ff00", "prompt": "bold green"}),
    multiline=False,
)

# ─── Streaming Function ─────────────────────────────────────────────────────────
def stream_graph_updates(graph, user_input: str):
    markdown_buffer = ""
    json_blobs: list[str] = []

    def render_panel():
        ts = datetime.now().strftime("%H:%M:%S")
        title = f"[dim]{ts}[/dim] [bold green]AI Response[/bold green]"
        return Panel(Markdown(markdown_buffer), title=title, border_style="cyan")

    try:
        with Live(Spinner("dots", text="Thinking…"), console=console, refresh_per_second=12) as live:
            for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    chunk = value["messages"][-1].content
                    stripped = chunk.strip()
                    try:
                        json.loads(stripped)
                    except json.JSONDecodeError:
                        markdown_buffer += chunk
                        live.update(render_panel())
                    else:
                        json_blobs.append(stripped)
    except KeyboardInterrupt:
        console.print("\n[bold red]⏹ Aborted current response.[/bold red]")
        return

    if json_blobs:
        console.rule("[bold blue]Tool-Call JSON Outputs")
        for blob in json_blobs:
            try:
                console.print(JSON(blob, indent=2))
            except Exception:
                console.print(Syntax(blob, "json", line_numbers=False))


# ─── Main Loop ──────────────────────────────────────────────────────────────────
def interact_with_graph(graph):
    console.print(
        Panel(
            "[bold magenta]LangGraph CLI[/bold magenta]\n"
            "Type [green]'exit'[/green] or [green]'help'[/green] to quit",
            title="LangGraph Chat",
            style="bold magenta"
        )
    )

    while True:
        # 1) Prompt for user input
        try:
            user_input = session.prompt(HTML("<prompt>You:</prompt> ")).strip()
        except KeyboardInterrupt:
            console.print("[bold yellow]Input cancelled—sending final goodbye…[/bold yellow]")
            stream_graph_updates(graph, "Goodbye")
            break

        cmd = user_input.lower()

        # 2) Handle control commands
        if cmd in {"exit", "quit", "q"}:
            console.print("[bold yellow]Exit requested—sending final goodbye…[/bold yellow]")
            stream_graph_updates(graph, "Goodbye")
            break

        if cmd == "help":
            # Use AI to generate dynamic help info, including listing tools and context
            help_prompt = (
                "Please provide a prompt-oriented user perspective list of available tools and relevant usage instructions."
                "for this LangGraph environment. No code leak."
            )
            try:
                stream_graph_updates(graph, help_prompt)
            except Exception as e:
                console.print(Panel(f"[bold red]Error fetching help:[/] {e}", style="red"))
            continue

        # 3) Echo the user's message
        timestamp = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{timestamp}[/] [bold cyan]You:[/] {user_input}")

        # 4) Stream the AI response
        try:
            stream_graph_updates(graph, user_input)
        except Exception as e:
            console.print(Panel(f"[bold red]Error during stream:[/] {e}", style="red"))
            continue

