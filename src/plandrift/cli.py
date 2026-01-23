"""Typer CLI for the travel planning agent."""

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.theme import Theme

from .agent import TravelAgent
from .models import Phase
from dotenv import load_dotenv


load_dotenv()

# Custom theme for the CLI
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "danger": "bold red",
        "success": "bold green",
        "phase": "bold magenta",
    }
)

console = Console(theme=custom_theme)
app = typer.Typer(
    name="plandrift",
    help="A constraint-first travel planning CLI agent",
    add_completion=False,
)


def print_phase_header(phase: Phase) -> None:
    """Print a styled phase header."""
    phase_names = {
        Phase.CLARIFICATION: "Phase 1: Clarification",
        Phase.FEASIBILITY: "Phase 2: Feasibility Check",
        Phase.ASSUMPTIONS: "Phase 3: Assumptions",
        Phase.PLANNING: "Phase 4: Plan Generation",
        Phase.REFINEMENT: "Phase 5: Refinement",
    }
    name = phase_names.get(phase, str(phase.value))
    console.print(f"\n[phase]━━━ {name} ━━━[/phase]\n")


def print_response(text: str) -> None:
    """Print agent response with markdown formatting."""
    md = Markdown(text)
    console.print(md)


def on_web_search(query: str) -> None:
    """Callback when a web search is performed."""
    console.print(f"  [dim]🔍 Searching: {query}[/dim]")


def run_conversation(agent: TravelAgent) -> None:
    """Run the main conversation loop."""
    while True:
        current_phase = agent.state.phase

        if current_phase == Phase.CLARIFICATION:
            # Already showed questions, get answers
            print_phase_header(Phase.CLARIFICATION)
            console.print("[info]Please answer all questions:[/info]\n")
            answers = Prompt.ask("Your answers")

            if not answers.strip():
                console.print("[warning]Please provide answers to continue.[/warning]")
                continue

            console.print()
            console.print("[info]Researching current travel conditions...[/info]")
            response, has_high_risk = agent.process_clarification(answers)

            print_phase_header(Phase.FEASIBILITY)
            print_response(response)

            if has_high_risk:
                proceed = Confirm.ask("\nProceed despite high risk?", default=False)
                response = agent.confirm_proceed(proceed)
                if not proceed:
                    print_response(response)
                    continue
                print_phase_header(Phase.ASSUMPTIONS)
                print_response(response)
            else:
                # Auto-proceed to assumptions
                with console.status("[info]Preparing assumptions...[/info]"):
                    response = agent.proceed_to_assumptions()
                print_phase_header(Phase.ASSUMPTIONS)
                print_response(response)

            # Now in assumptions phase
            agent.state.phase = Phase.ASSUMPTIONS

        elif current_phase == Phase.ASSUMPTIONS:
            confirmed = Confirm.ask("\nConfirm these assumptions?", default=True)

            if not confirmed:
                adjustments = Prompt.ask("What adjustments do you need?")
                with console.status("[info]Adjusting assumptions...[/info]"):
                    response = agent.confirm_assumptions(False, adjustments)
                print_response(response)
                continue

            console.print()
            console.print("[info]Researching and generating your travel plan...[/info]")
            response = agent.confirm_assumptions(True)

            print_phase_header(Phase.PLANNING)
            print_response(response)

        elif current_phase == Phase.PLANNING:
            # Should not reach here normally, but handle it
            pass

        elif current_phase == Phase.REFINEMENT:
            console.print()
            choice = Prompt.ask(
                "Enter refinement number (1-5) or 'done' to finish",
                default="done",
            )

            if choice.lower() == "done":
                console.print(
                    Panel(
                        "[success]Your travel plan is ready! Safe travels! 🌍[/success]",
                        title="Plan Complete",
                        border_style="green",
                    )
                )
                break

            refinement_map = {
                "1": "Make it safer",
                "2": "Make it faster",
                "3": "Reduce travel hours",
                "4": "Increase comfort",
                "5": "Change base location",
            }

            if choice in refinement_map:
                with console.status(
                    f"[info]Refining: {refinement_map[choice]}...[/info]"
                ):
                    response = agent.refine_plan(refinement_map[choice])
                print_response(response)
            else:
                console.print("[warning]Invalid choice. Enter 1-5 or 'done'.[/warning]")

        else:
            break


@app.command()
def plan(
    trip: Optional[str] = typer.Argument(
        None,
        help="Trip description (e.g., 'Mumbai to SF', 'Delhi to Ladakh')",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
        envvar="OPENAI_API_KEY",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="OpenAI model to use",
    ),
) -> None:
    """Start a constraint-first travel planning session."""
    console.print(
        Panel(
            "[bold]Plandrift[/bold] - Constraint-First Travel Planning\n\n"
            "I'll help you plan your trip by:\n"
            "1. Asking clarifying questions\n"
            "2. Checking feasibility and risks\n"
            "3. Confirming assumptions\n"
            "4. Generating a detailed itinerary\n"
            "5. Refining based on your preferences",
            title="Welcome",
            border_style="cyan",
        )
    )

    # Get origin and destination
    origin: Optional[str] = None
    destination: Optional[str] = None

    if trip:
        # Try to parse "from X to Y" or "X to Y" format
        trip_lower = trip.lower()
        if " to " in trip_lower:
            parts = trip.split(" to ", 1) if " to " in trip else trip.split(" TO ", 1)
            # Handle "from X to Y" format
            if parts[0].lower().startswith("from "):
                origin = parts[0][5:].strip()
            else:
                origin = parts[0].strip()
            destination = parts[1].strip()
        else:
            # Just destination provided, ask for origin
            destination = trip.strip()

    # Ask for missing information
    if not origin:
        origin = Prompt.ask("\n[info]Where are you traveling from?[/info]")
    if not destination:
        destination = Prompt.ask("[info]Where do you want to go?[/info]")

    if not origin.strip() or not destination.strip():
        console.print("[danger]Both origin and destination are required.[/danger]")
        raise typer.Exit(1)

    # Initialize agent with search callback
    try:
        agent = TravelAgent(api_key=api_key, model=model, on_search=on_web_search)
    except ValueError as e:
        console.print(f"[danger]{e}[/danger]")
        console.print("\n[info]Set your API key:[/info]")
        console.print("  export OPENAI_API_KEY='your-key-here'")
        console.print("  # or use --api-key flag")
        raise typer.Exit(1)

    # Start conversation
    console.print(f"\n[success]Planning trip: {origin} → {destination}[/success]\n")

    with console.status("[info]Preparing clarification questions...[/info]"):
        response = agent.start(origin, destination)

    print_response(response)

    # Run conversation loop
    try:
        run_conversation(agent)
    except KeyboardInterrupt:
        console.print("\n\n[warning]Planning session ended.[/warning]")
        raise typer.Exit(0)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__

    console.print(f"plandrift version {__version__}")


if __name__ == "__main__":
    app()
