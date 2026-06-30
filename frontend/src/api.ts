export type HealthStatus = {
  status: string;
  model_loaded: boolean;
  weights_path: string;
  device: string;
};

export type AnalysisResult = {
  study_id: number;
  predicted_class: string;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  tumor_detected: boolean;
  heatmap_base64?: string;
  model_version: string;
  limitations: string[];
};

export type StudyRow = {
  id: number;
  study_ref: string;
  radiologist_id: string;
  predicted_label: string;
  confidence: number;
  created_at: string;
};

export type AuditRow = {
  id: number;
  actor: string;
  action: string;
  detail: string | null;
  created_at: string;
};

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function analyzeScan(form: FormData): Promise<AnalysisResult> {
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function fetchStudies(): Promise<StudyRow[]> {
  const res = await fetch("/api/studies");
  if (!res.ok) throw new Error("Failed to load studies");
  return res.json();
}

export async function fetchAudit(): Promise<AuditRow[]> {
  const res = await fetch("/api/audit");
  if (!res.ok) throw new Error("Failed to load audit log");
  return res.json();
}
