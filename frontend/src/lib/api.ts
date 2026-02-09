/**
 * Plandrift API client.
 *
 * Handles authentication (JWT token management) and all trip-planning
 * endpoints.  Tokens are stored in localStorage so they survive page
 * reloads but are cleared on logout.
 */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://plandrift.onrender.com";
const API_PREFIX = `${API_BASE}/api`;

export function getGoogleLoginUrl(): string {
  return `${API_PREFIX}/auth/google/login`;
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

const TOKEN_KEY = "plandrift_access_token";
const REFRESH_KEY = "plandrift_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_KEY);
}

export function setTokens(access: string, refresh: string): void {
  localStorage.setItem(TOKEN_KEY, access);
  localStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuthUser {
  id: string;
  email: string;
  name: string | null;
  picture_url?: string | null;
  user_type?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  user: AuthUser;
}

export interface AgentResponse {
  trip_id: string | null;
  version_id: string | null;
  phase: string;
  message: string;
  has_high_risk: boolean;
}

export interface TripSummary {
  id: string;
  origin: string;
  destination: string;
  status: string | null;
  phase: string | null;
  created_at: string;
  updated_at: string;
}

export interface TripVersionDetail {
  id: string;
  trip_id: string;
  version_number: number;
  status: string;
  phase: string;
  constraints_json: Record<string, unknown> | null;
  risk_assessment_json: Record<string, unknown> | null;
  assumptions_json: Record<string, unknown> | null;
  plan_json: Record<string, unknown> | null;
  budget_breakdown_json: Record<string, unknown> | null;
  days_json: Record<string, unknown>[] | null;
  created_at: string;
  updated_at: string;
}

export interface TripDetail {
  id: string;
  user_id: string;
  origin: string;
  destination: string;
  created_at: string;
  updated_at: string;
  latest_version: TripVersionDetail | null;
}

export interface TripWithVersions {
  id: string;
  user_id: string;
  origin: string;
  destination: string;
  created_at: string;
  updated_at: string;
  versions: {
    id: string;
    version_number: number;
    status: string;
    phase: string;
    created_at: string;
  }[];
}

export type StreamMeta = {
  trip_id: string | null;
  version_id: string | null;
  phase: string;
  has_high_risk: boolean;
};

export type StreamEvent =
  | { type: "meta"; data: StreamMeta }
  | { type: "delta"; data: string }
  | { type: "status"; data: string }
  | { type: "done" };

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const token = getAccessToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_PREFIX}${path}`, {
    ...options,
    headers,
  });

  // Handle 401 — try token refresh once
  if (res.status === 401 && getRefreshToken()) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      // Retry original request with new token
      headers["Authorization"] = `Bearer ${getAccessToken()}`;
      const retry = await fetch(`${API_PREFIX}${path}`, {
        ...options,
        headers,
      });
      if (retry.ok) {
        if (retry.status === 204) return undefined as T;
        return retry.json();
      }
      const retryBody = await retry.json().catch(() => ({}));
      throw new ApiError(retry.status, retryBody.detail || "Request failed");
    }
    // Refresh failed — clear tokens and throw
    clearTokens();
    throw new ApiError(401, "Session expired. Please sign in again.");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail || `Request failed (${res.status})`);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

async function streamFetch(
  path: string,
  body: Record<string, unknown>,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const token = getAccessToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_PREFIX}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new ApiError(res.status, err.detail || "Streaming request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  async function* iterator(): AsyncGenerator<StreamEvent, void, void> {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let eventType = "message";
        let data = "";
        for (const line of rawEvent.split("\n")) {
          if (line.startsWith("event:")) {
            eventType = line.replace("event:", "").trim();
          } else if (line.startsWith("data:")) {
            data += line.replace("data:", "").trim();
          }
        }

        if (eventType === "meta") {
          yield { type: "meta", data: JSON.parse(data) as StreamMeta };
        } else if (eventType === "delta") {
          const payload = JSON.parse(data) as { text: string };
          yield { type: "delta", data: payload.text };
        } else if (eventType === "status") {
          const payload = JSON.parse(data) as { text: string };
          yield { type: "status", data: payload.text };
        } else if (eventType === "done") {
          yield { type: "done" };
        }
      }
    }
  }

  return iterator();
}

async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_PREFIX}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;

    const data: AuthResponse = await res.json();
    setTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(
  email: string,
  password: string,
): Promise<AuthResponse> {
  const data = await apiFetch<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function register(
  email: string,
  password: string,
  name?: string,
): Promise<AuthResponse> {
  const data = await apiFetch<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function logout(): Promise<void> {
  const refreshToken = getRefreshToken();
  try {
    await apiFetch("/auth/logout", {
      method: "POST",
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } finally {
    clearTokens();
  }
}

export async function getProfile(): Promise<AuthUser> {
  return apiFetch<AuthUser>("/auth/profile");
}

// ---------------------------------------------------------------------------
// Trip conversation (stateful, phase-by-phase)
// ---------------------------------------------------------------------------

export async function startTrip(prompt: string): Promise<AgentResponse> {
  return apiFetch<AgentResponse>("/trips/start", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export async function startTripStream(
  prompt: string,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  return streamFetch("/trips/start/stream", { prompt });
}

export async function clarifyTrip(
  tripId: string,
  answers: string,
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/trips/${tripId}/clarify`, {
    method: "POST",
    body: JSON.stringify({ answers }),
  });
}

export async function clarifyTripStream(
  tripId: string,
  answers: string,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  return streamFetch(`/trips/${tripId}/clarify/stream`, { answers });
}

export async function proceedTrip(
  tripId: string,
  proceed: boolean,
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/trips/${tripId}/proceed`, {
    method: "POST",
    body: JSON.stringify({ proceed }),
  });
}

export async function proceedTripStream(
  tripId: string,
  proceed: boolean,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  return streamFetch(`/trips/${tripId}/proceed/stream`, { proceed });
}

export async function confirmAssumptions(
  tripId: string,
  confirmed: boolean,
  modifications?: string,
  additionalInterests?: string,
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/trips/${tripId}/assumptions`, {
    method: "POST",
    body: JSON.stringify({
      confirmed,
      modifications: modifications || null,
      additional_interests: additionalInterests || null,
    }),
  });
}

export async function confirmAssumptionsStream(
  tripId: string,
  confirmed: boolean,
  modifications?: string,
  additionalInterests?: string,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  return streamFetch(`/trips/${tripId}/assumptions/stream`, {
    confirmed,
    modifications: modifications || null,
    additional_interests: additionalInterests || null,
  });
}

export async function refineTrip(
  tripId: string,
  refinementType: string,
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/trips/${tripId}/refine`, {
    method: "POST",
    body: JSON.stringify({ refinement_type: refinementType }),
  });
}

export async function refineTripStream(
  tripId: string,
  refinementType: string,
): Promise<AsyncGenerator<StreamEvent, void, void>> {
  return streamFetch(`/trips/${tripId}/refine/stream`, {
    refinement_type: refinementType,
  });
}

// ---------------------------------------------------------------------------
// Trip CRUD (stateless)
// ---------------------------------------------------------------------------

export async function getTrips(): Promise<TripSummary[]> {
  return apiFetch<TripSummary[]>("/trips");
}

export async function getTrip(tripId: string): Promise<TripDetail> {
  return apiFetch<TripDetail>(`/trips/${tripId}`);
}

export async function getTripVersions(
  tripId: string,
): Promise<TripWithVersions> {
  return apiFetch<TripWithVersions>(`/trips/${tripId}/versions`);
}

export async function deleteTrip(tripId: string): Promise<void> {
  return apiFetch<void>(`/trips/${tripId}`, { method: "DELETE" });
}
