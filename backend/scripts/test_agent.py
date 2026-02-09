#!/usr/bin/env python3
"""CLI tool for testing the TravelAgent interactively.

Usage:
    cd backend
    uv run python scripts/test_agent.py

Make sure OPENROUTER_API_KEY is set in your environment or .env file.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent import TravelAgent
from app.agent.models import Phase

load_dotenv()


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_separator():
    print("\n" + "=" * 80 + "\n")


def print_agent(text: str):
    """Print an agent response."""
    print(f"\n🤖 Agent:\n{'-' * 80}\n{text}\n{'-' * 80}")


def print_status(message: str):
    """Print a status / processing message."""
    print(f"\n⏳ {message}")


def ask(prompt: str = "You: ") -> str:
    """Prompt the user for input. Ctrl-C exits."""
    try:
        return input(f"\n{prompt}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Goodbye!")
        sys.exit(0)


def on_search(query: str):
    """Callback printed every time the agent fires a web search."""
    print(f"  🔍 Searching: {query}")


# ── Phase handlers ───────────────────────────────────────────────────────────

def do_clarification(agent: TravelAgent) -> tuple[str, bool]:
    """Collect the user's answers and run clarification → feasibility."""
    print("\n📋 Answer the questions above:")
    answers = ask("You: ")
    while not answers:
        print("⚠️  Please provide at least one answer.")
        answers = ask("You: ")

    print_status("Analyzing your answers and checking feasibility...")
    return agent.process_clarification(answers)   # (risk_text, has_high_risk)


def do_high_risk_confirm(agent: TravelAgent) -> str | None:
    """Ask the user whether to proceed despite high risk.

    Returns the assumptions text if user proceeds, None otherwise.
    """
    while True:
        choice = ask("Do you want to proceed anyway? (yes/no): ").lower()
        if choice in ("yes", "y"):
            # confirm_proceed(True) internally generates assumptions
            return agent.confirm_proceed(True)
        elif choice in ("no", "n"):
            response = agent.confirm_proceed(False)
            print_agent(response)
            return None
        else:
            print("Please enter 'yes' or 'no'.")


def do_assumptions(agent: TravelAgent) -> str | None:
    """Let the user confirm or modify assumptions.

    Returns the plan text once confirmed, or None if the user bails.
    """
    print("\n📝 Review the assumptions above.")

    user_input = ask("Type 'yes' to continue, or describe any modifications: ")

    if not user_input:
        print("⚠️  Please type 'yes' to continue or describe your modifications.")
        user_input = ask("Type 'yes' to continue, or describe any modifications: ")

    if user_input.lower() in (
        "yes", "y", "ok", "continue", "proceed", "looks good", "confirm",
    ):
        print_status("Generating your travel plan (this may take a minute)...")
        return agent.confirm_assumptions(confirmed=True)

    elif user_input:
        print_status("Incorporating your preferences and generating travel plan...")
        # Modifications are incorporated and planning proceeds directly
        return agent.confirm_assumptions(
            confirmed=False, modifications=user_input,
        )

    return None


def do_refinement(agent: TravelAgent):
    """Let the user refine the plan or accept it."""
    print("\n🔧 You can refine your plan or accept it.")

    while True:
        choice = ask(
            "Refinement options:\n"
            "  1. Make it safer\n"
            "  2. Make it faster\n"
            "  3. Reduce travel hours\n"
            "  4. Increase comfort\n"
            "  5. Change base location\n"
            "  6. Accept plan\n"
            "\nYour choice (1-6 or 'done'): "
        )

        if choice in ("6", "done", "accept"):
            print("\n✅ Plan accepted!")
            return

        refinement_map = {
            "1": "safer",
            "2": "faster",
            "3": "reduce travel hours",
            "4": "increase comfort",
            "5": "change base location",
        }
        if choice in refinement_map:
            print_status(f"Refining plan: {refinement_map[choice]}...")
            response = agent.refine_plan(refinement_map[choice])
            print_agent(response)
        else:
            print("Please enter a number between 1-6.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("🧳 Travel Agent CLI — Testing Mode")
    print("=" * 80)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n⚠️  OPENROUTER_API_KEY not set.")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'\n")

    agent = TravelAgent(api_key=api_key, on_search=on_search)

    # ── 1. Get the initial prompt ────────────────────────────────────────
    print("\nDescribe your trip (e.g. 'Plan a trip from Mumbai to Japan'):")
    print_separator()

    initial_prompt = ask("You: ")
    while not initial_prompt:
        print("⚠️  Please describe your trip.")
        initial_prompt = ask("You: ")

    print_status("Understanding your request...")
    response = agent.start(initial_prompt)
    print_agent(response)

    # ── 2. Ensure we have origin + destination ───────────────────────────
    while not agent.state.origin or not agent.state.destination:
        follow_up = ask("You: ")
        if not follow_up:
            print("❌ Cannot proceed without origin and destination.")
            sys.exit(1)
        print_status("Understanding your request...")
        response = agent.start(f"{initial_prompt}. {follow_up}")
        print_agent(response)

    # ── 3. Walk through each phase ───────────────────────────────────────
    while True:
        phase = agent.state.phase

        # ─── CLARIFICATION ───────────────────────────────────────────
        if phase == Phase.CLARIFICATION:
            risk_text, has_high_risk = do_clarification(agent)
            print_agent(risk_text)

            if has_high_risk:
                # User must decide whether to proceed
                assumptions_text = do_high_risk_confirm(agent)
                if assumptions_text is None:
                    break  # user chose not to proceed
                print_agent(assumptions_text)
            else:
                # No high risk — generate assumptions automatically
                print_status("Generating assumptions...")
                assumptions_text = agent._generate_assumptions()
                print_agent(assumptions_text)

        # ─── ASSUMPTIONS ─────────────────────────────────────────────
        elif phase == Phase.ASSUMPTIONS:
            plan_text = do_assumptions(agent)
            if plan_text:
                print_agent(plan_text)

        # ─── REFINEMENT ──────────────────────────────────────────────
        elif phase == Phase.REFINEMENT:
            do_refinement(agent)
            break

        # ─── ANYTHING ELSE ───────────────────────────────────────────
        else:
            print(f"\n✅ Done! (phase: {phase})")
            break

    print_separator()
    print("🎉 Session complete. Run this script again to test another trip.")


if __name__ == "__main__":
    main()
