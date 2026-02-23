# AGENTS.md - Development Guide for Agentic Coding Agents

This document provides essential information for agentic coding agents working with the Planfirst codebase. It covers build/lint/test commands, code style guidelines, and development workflows.

## Codebase Structure

```
planfirst/
├── src/planfirst/     # CLI tool (Python)
├── backend/           # FastAPI API (Python)
├── frontend/          # Next.js frontend (TypeScript)
└── AGENTS.md          # This file
```

## Build/Lint/Test Commands

### Python (Backend & CLI)

**Package Management:**
```bash
# Install/Update dependencies (always use uv)
uv sync

# Add new dependency
uv add fastapi

# Remove dependency
uv remove fastapi
```

**Development Server:**
```bash
# Backend development server
cd backend
uv run uvicorn app.main:app --reload --port 8000

# CLI development
uv run planfirst plan "destination"

# Or directly
uv run python -m planfirst.cli plan "destination"
```

**Database Migrations:**
```bash
# Create migration
cd backend
uv run alembic revision --autogenerate -m "description"

# Run migrations
uv run alembic upgrade head
```

**Linting:**
```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .
```

**Testing:**
```bash
# Run all tests (when available)
uv run pytest

# Run specific test file
uv run pytest tests/test_specific.py

# Run single test function
uv run pytest tests/test_file.py::test_function_name

# Run tests with coverage
uv run pytest --cov=app --cov-report=html
```

### TypeScript (Frontend)

**Package Management:**
```bash
# Install dependencies (always use pnpm)
cd frontend
pnpm install

# Add new dependency
pnpm add react

# Add dev dependency
pnpm add -D typescript
```

**Development Server:**
```bash
# Frontend development server
cd frontend
pnpm dev
```

**Linting:**
```bash
# Check for linting issues
pnpm lint

# Fix linting issues
pnpm lint --fix
```

**Testing:**
```bash
# Run frontend tests (when configured)
pnpm test

# Run specific test file
pnpm test -- some.test.ts

# Run tests in watch mode
pnpm test -- --watch
```

## Code Style Guidelines

### Python Standards

**Imports:**
1. Standard library imports
2. Third-party imports
3. Local project imports
4. Separate with blank lines

```python
import os
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.security import hash_password
from app.db.models import User
```

**Formatting:**
- Use ruff format (Black-style formatting)
- Max line length: 88 characters
- Indentation: 4 spaces
- Files should end with a newline

**Naming Conventions:**
- snake_case for variables, functions, modules
- PascalCase for classes
- UPPER_CASE for constants
- Private members prefixed with underscore (_internal_var)

**Type Hints:**
Always include type hints for parameters and return values:

```python
def calculate_travel_budget(destination: str, days: int, budget_level: str) -> dict:
    return {"total": 1000, "daily_average": 100}
```

**Pydantic Models:**
Use Pydantic for all data validation:

```python
from pydantic import BaseModel, Field

class TripRequest(BaseModel):
    origin: str = Field(..., description="Starting location")
    destination: str = Field(..., description="Travel destination")
    duration_days: int = Field(..., gt=0, description="Trip duration in days")
    
    class Config:
        from_attributes = True
```

**SQLAlchemy 2.0 Patterns:**
Use modern SQLAlchemy 2.0 syntax with type hints:

```python
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
```

**Async/Await:**
Database operations should be async:

```python
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()
```

### TypeScript Standards

**Imports:**
Group imports logically:
1. External packages
2. Internal packages
3. Relative imports
4. Type imports

```typescript
// External
import { useState, useEffect } from 'react';
import { z } from 'zod';

// Internal
import { api } from '@/lib/api';

// Relative
import { Button } from './button';
import type { Trip } from './types';
```

**Component Structure:**
Use functional components with TypeScript interfaces:

```typescript
interface TripFormProps {
  origin: string;
  destination: string;
  onSubmit: (data: TripFormData) => void;
}

export function TripForm({ origin, destination, onSubmit }: TripFormProps) {
  // Component implementation
}
```

**Naming Conventions:**
- PascalCase for components (`TripForm`)
- camelCase for variables and functions (`handleSubmit`)
- UPPER_CASE for constants (`MAX_DAYS`)
- Interfaces without 'I' prefix

### Error Handling

**Python:**
Handle errors gracefully and return appropriate HTTP status codes:

```python
from fastapi import HTTPException

async def get_trip_or_404(db: AsyncSession, trip_id: int) -> Trip:
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip
```

**TypeScript:**
Use proper error boundaries and async/await patterns:

```typescript
try {
  const response = await api.createTrip(tripData);
  // Handle success
} catch (error) {
  if (error instanceof ApiError) {
    // Handle API-specific errors
  } else {
    // Handle unexpected errors
  }
}
```

## Architectural Principles (SOLID)

To maintain a scalable and maintainable codebase, we follow SOLID principles:

1.  **Single Responsibility (SRP)**: Each class/module should have one reason to change. 
    - *Example*: Keep database models in `models.py` and business logic in `services/`.
2.  **Open/Closed (OCP)**: Software entities should be open for extension but closed for modification. 
    - *Example*: Use the `Phase` system in the agent to add new planning stages without modifying the core `TravelAgent` logic.
3.  **Liskov Substitution (LSP)**: Subtypes must be substitutable for their base types.
    - *Example*: Any search provider (Flight, Hotel) should adhere to the same interface.
4.  **Interface Segregation (ISP)**: Clients should not be forced to depend on methods they do not use.
    - *Example*: Use granular Pydantic schemas for different API endpoints rather than one giant "Trip" object.
5.  **Dependency Inversion (DIP)**: Depend on abstractions, not concretions.
    - *Example*: FastAPI's `Depends` system for injecting database sessions and authenticated users.

## Project-Specific Requirements

### Package Managers
1. Python: Use `uv` exclusively (never pip/npm)
2. Node.js: Use `pnpm` exclusively (never npm/yarn)

### Libraries
1. Search: Use `ddgs` (NOT duckduckgo-search)
2. Auth: Use `better-auth` for frontend (NOT NextAuth)
3. Database: SQLAlchemy 2.0 patterns with asyncpg
4. Validation: Pydantic everywhere in Python

### Environment Context
Always include current date/year in web search queries for current information:

```python
from datetime import datetime
current_year = datetime.now().year
search_query = f"Ladakh travel advisories {current_year}"
```

## Common Development Workflows

1. **Adding a new API endpoint:**
   - Create schema in `backend/app/schemas/`
   - Add route in `backend/app/api/v1/`
   - Implement business logic in `backend/app/services/`
   - Update database models if needed in `backend/app/db/models.py`

2. **Adding a new CLI command:**
   - Add command function in `src/planfirst/cli.py`
   - Implement logic in separate modules under `src/planfirst/`
   - Add required schemas in `src/planfirst/models.py`

3. **Adding a new frontend component:**
   - Create component file in `frontend/src/components/`
   - Add TypeScript interfaces for props
   - Follow existing styling patterns with Tailwind

## Troubleshooting

**Common Issues:**
1. Missing dependencies: Run `uv sync` in backend or `pnpm install` in frontend
2. Database connection: Ensure PostgreSQL is running and .env is configured
3. Type errors: Check pydantic models and SQLAlchemy mappings
4. Import errors: Verify file paths and package structure

**Debugging Steps:**
1. Check error messages in console
2. Verify all required environment variables are set
3. Confirm all dependencies are installed
4. Check file permissions and paths
5. Look at recent commits for breaking changes