import type {
  HealthResponse,
  SessionAnalyzeResponse,
  SessionFeatureVector,
  TrialPredictResponse
} from "../types/api";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const hasBody = Boolean(init?.body);
  const headers = new Headers(init?.headers ?? {});
  if (hasBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // Fall back to status text when JSON detail is unavailable.
      detail = response.statusText || detail;
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function predictTrial(subjectId: number): Promise<TrialPredictResponse> {
  const dummyEpoch = Array.from({ length: 22 }, () =>
    Array.from({ length: 1501 }, () => Number((Math.random() * 0.01).toFixed(6)))
  );
  return request<TrialPredictResponse>(`/predict/trial?subject_id=${subjectId}`, {
    method: "POST",
    body: JSON.stringify({
      epoch: dummyEpoch,
      y_true: Math.random() > 0.5 ? "left_hand" : "right_hand"
    })
  });
}

export async function analyzeSession(
  features: SessionFeatureVector
): Promise<SessionAnalyzeResponse> {
  return request<SessionAnalyzeResponse>("/analyze/session", {
    method: "POST",
    body: JSON.stringify({ features })
  });
}

export { API_BASE, ApiError };
