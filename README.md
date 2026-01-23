# Plandrift

A constraint-first travel planning CLI agent powered by OpenAI with real-time web search.

Unlike typical travel planners that immediately generate generic itineraries, Plandrift **thinks before planning** by:

1. **Clarifying constraints** - Asks targeted questions about season, duration, budget, and comfort level
2. **Checking feasibility** - Searches the web for current travel advisories, weather, and conditions
3. **Confirming assumptions** - Makes all planning assumptions explicit before proceeding
4. **Generating detailed plans** - Researches current info and creates day-by-day itineraries with reasoning
5. **Offering refinements** - Allows adjustments for safety, speed, comfort, and more

## Features

- **Real-time web search** - Automatically searches for current travel advisories, visa requirements, weather conditions, and infrastructure updates
- **Constraint-first approach** - Refuses to plan until it understands your constraints
- **Risk assessment** - Evaluates season, altitude, accessibility, and infrastructure risks
- **Structured planning** - Day-by-day itineraries with travel times, accommodations, and reasoning

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# Clone and install
cd plandrift
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Start a planning session

```bash
# Interactive mode (will ask for origin and destination)
uv run plandrift plan

# With origin and destination
uv run plandrift plan "Mumbai to Ladakh"

# Alternative formats
uv run plandrift plan "from Delhi to Iceland"

# With custom model
uv run plandrift plan "SF to Tokyo" --model gpt-4o-mini
```

### Example Session

```
$ uv run plandrift plan "Mumbai to Ladakh"

Planning trip: Mumbai → Ladakh

━━━ Phase 1: Clarification ━━━

1. What month or season are you planning to travel?
2. How many total days (including travel)?
3. Solo or group travel?
4. What's your budget level (budget/mid-range/luxury)?
5. Comfort with rough conditions (low/medium/high)?

Your answers: > July, 10 days, solo, mid-range, medium comfort

━━━ Phase 2: Feasibility Check ━━━

Researching current travel conditions...
  🔍 Searching: Ladakh travel advisory 2024
  🔍 Searching: Ladakh weather July monsoon conditions
  🔍 Searching: Leh Manali highway status

🟡 Season & Weather: MEDIUM
🟢 Route Accessibility: LOW
🟡 Altitude & Health: MEDIUM
🟡 Infrastructure: MEDIUM

Warnings:
⚠️ July is monsoon season - expect road closures and landslides
⚠️ Altitude ranges from 3,500m to 5,600m - acclimatization required

━━━ Phase 3: Assumptions ━━━

• 10 days total including Delhi flights
• Mid-range hotels and guesthouses
• Solo traveler comfortable with shared transport
...

━━━ Phase 4: Plan Generation ━━━

Day 1: Arrive Delhi
  • Arrive and rest
  🏨 Stay: Hotel in Paharganj
  💡 Why: Buffer day for flight delays and rest before altitude

Day 2: Fly to Leh
  • Morning flight to Leh (1 hour)
  • Complete rest for acclimatization
  🚗 Travel: 1 hour flight
  🏨 Stay: Guesthouse in Leh
  💡 Why: Critical acclimatization day - no exertion
...
```

## Commands

| Command | Description |
|---------|-------------|
| `plandrift plan [DESTINATION]` | Start a travel planning session |
| `plandrift version` | Show version information |
| `plandrift --help` | Show help |

## Options

| Option | Description |
|--------|-------------|
| `--api-key, -k` | OpenAI API key (or use `OPENAI_API_KEY` env var) |
| `--model, -m` | OpenAI model to use (default: `gpt-4o`) |

## How It Works

### Phase 1: Clarification
The agent refuses to plan until it understands your constraints. It asks exactly 5 questions covering:
- Season/month of travel
- Total duration
- Solo vs group
- Budget level
- Comfort tolerance

### Phase 2: Feasibility Check
Risk assessment across four dimensions:
- **Season & Weather**: Monsoons, winter closures, extreme temperatures
- **Route Accessibility**: Road conditions, permits, seasonal closures
- **Altitude & Health**: Acclimatization needs, medical risks
- **Infrastructure**: Connectivity, accommodation availability

High-risk trips require explicit user confirmation.

### Phase 3: Assumptions
All planning assumptions are made explicit:
- Inferred preferences
- Default choices
- Uncertain assumptions marked with `[?]`

User confirms or adjusts before planning proceeds.

### Phase 4: Plan Generation
Day-by-day itinerary with:
- Specific activities and timings
- Realistic travel times
- Accommodation recommendations
- Reasoning for each day's structure
- Buffer days for weather/acclimatization

### Phase 5: Refinement
Post-generation adjustments:
1. Make it safer
2. Make it faster
3. Reduce travel hours
4. Increase comfort
5. Change base location

## Development

```bash
# Install dev dependencies
uv sync

# Run directly
uv run python -m plandrift.cli plan

# Install as editable
uv pip install -e .
plandrift plan
```

## License

MIT
