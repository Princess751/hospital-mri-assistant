import { useCallback, useEffect, useState } from "react";
import {
  analyzeScan,
  fetchAudit,
  fetchHealth,
  fetchStudies,
  type AnalysisResult,
  type AuditRow,
  type HealthStatus,
  type StudyRow,
} from "./api";

type Tab = "analyze" | "studies" | "audit" | "about";
const TABS: { id: Tab; label: string }[] = [
  { id: "analyze", label: "Analyze" },
  { id: "studies", label: "Studies" },
  { id: "audit", label: "Audit" },
  { id: "about", label: "About" },
];

const fmt = (iso: string) => iso.slice(0, 19).replace("T", " ");

export default function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [radId, setRadId] = useState("");
  const [studyRef, setStudyRef] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; kind: "" | "ok" | "error" }>({ text: "", kind: "" });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [studies, setStudies] = useState<StudyRow[]>([]);
  const [audit, setAudit] = useState<AuditRow[]>([]);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const loadStudies = useCallback(async () => {
    try { setStudies(await fetchStudies()); } catch { setStudies([]); }
  }, []);

  const loadAudit = useCallback(async () => {
    try { setAudit(await fetchAudit()); } catch { setAudit([]); }
  }, []);

  useEffect(() => {
    if (tab === "studies") loadStudies();
    if (tab === "audit") loadAudit();
  }, [tab, loadStudies, loadAudit]);

  function onFile(f: File | null) {
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setResult(null);
  }

  async function runAnalysis() {
    setMsg({ text: "", kind: "" });
    if (!radId.trim() || !studyRef.trim() || !file) {
      setMsg({ text: "Radiologist ID, study reference, and MRI file are required.", kind: "error" });
      return;
    }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("radiologist_id", radId.trim());
    fd.append("study_ref", studyRef.trim());
    fd.append("include_heatmap", heatmap ? "true" : "false");
    setBusy(true);
    try {
      const data = await analyzeScan(fd);
      setResult(data);
      setMsg({ text: `Analysis complete. Study ID ${data.study_id}`, kind: "ok" });
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header>
        <div className="wrap">
          <h1>Hospital MRI Assistant</h1>
          <div className="sub">
            AI decision-support for radiologists — classify brain MRI slices, view confidence scores,
            and inspect Grad-CAM attention maps. Final diagnosis remains with the clinician.
          </div>
          <div className="sub">
            {health?.model_loaded ? (
              <><span className="dot ok" />Model loaded — {health.device}</>
            ) : (
              <><span className="dot bad" />Model not ready — run setup via run.bat</>
            )}
          </div>
          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t.id} type="button" className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        {tab === "analyze" && (
          <section className="panel">
            <div className="banner">
              <strong>Clinical notice:</strong> Research and decision-support prototype only.
              Not cleared as a medical device. Requires radiologist review.
            </div>
            <h2>Analyze MRI scan</h2>
            <div className="grid two">
              <div>
                <div className="row">
                  <label className="field" htmlFor="rad">Radiologist ID</label>
                  <input id="rad" value={radId} onChange={(e) => setRadId(e.target.value)} placeholder="dr.smith" />
                </div>
                <div className="row">
                  <label className="field" htmlFor="ref">Study reference</label>
                  <input id="ref" value={studyRef} onChange={(e) => setStudyRef(e.target.value)} placeholder="STUDY-2026-0100" />
                </div>
                <div className="row">
                  <label className="field" htmlFor="file">MRI image</label>
                  <input id="file" type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
                </div>
                <div className="row">
                  <label><input type="checkbox" checked={heatmap} onChange={(e) => setHeatmap(e.target.checked)} /> Include Grad-CAM heatmap</label>
                </div>
                <button className="primary" type="button" disabled={busy} onClick={runAnalysis}>
                  {busy ? "Analyzing…" : "Run analysis"}
                </button>
                {msg.text && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
              </div>
              <div>
                <span className="field">Preview</span>
                {preview ? <img src={preview} className="preview" alt="MRI preview" /> : <div className="muted">No image selected.</div>}
              </div>
            </div>
            {result && (
              <div style={{ marginTop: "1.25rem" }}>
                <h2>Results</h2>
                <div className="grid two">
                  <div>
                    <div className={`pill ${result.tumor_detected ? "bad" : "ok"}`}>
                      {result.tumor_detected ? "Tumor suspected" : "No tumor class selected"}
                    </div>
                    <p style={{ margin: "0.75rem 0 0.25rem" }}>
                      {result.predicted_label} ({(result.confidence * 100).toFixed(1)}%)
                    </p>
                    <p className="muted" style={{ margin: 0 }}>Model: {result.model_version}</p>
                    <div style={{ marginTop: "1rem" }}>
                      {Object.entries(result.probabilities).map(([k, v]) => {
                        const pct = (v * 100).toFixed(1);
                        return (
                          <div key={k} className="prob-row">
                            <div className="prob-label"><span>{k}</span><span>{pct}%</span></div>
                            <div className="prob-bar"><div className="prob-fill" style={{ width: `${pct}%` }} /></div>
                          </div>
                        );
                      })}
                    </div>
                    <ul className="limitations">{result.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
                  </div>
                  <div>
                    <span className="field">Grad-CAM overlay</span>
                    {result.heatmap_base64 ? (
                      <img src={`data:image/png;base64,${result.heatmap_base64}`} className="heatmap" alt="Grad-CAM" />
                    ) : (
                      <div className="muted">Heatmap unavailable.</div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {tab === "studies" && (
          <section className="panel">
            <button className="ghost" type="button" onClick={loadStudies}>Refresh</button>
            <h2>Recent studies</h2>
            <table>
              <thead><tr><th>Time</th><th>Study</th><th>Radiologist</th><th>Finding</th><th>Confidence</th></tr></thead>
              <tbody>
                {studies.length === 0 ? (
                  <tr><td colSpan={5} className="muted">No studies yet.</td></tr>
                ) : studies.map((r) => (
                  <tr key={r.id}>
                    <td>{fmt(r.created_at)}</td><td>{r.study_ref}</td><td>{r.radiologist_id}</td>
                    <td>{r.predicted_label}</td><td>{(r.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {tab === "audit" && (
          <section className="panel">
            <button className="ghost" type="button" onClick={loadAudit}>Refresh</button>
            <h2>Audit trail</h2>
            <table>
              <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Detail</th></tr></thead>
              <tbody>
                {audit.length === 0 ? (
                  <tr><td colSpan={4} className="muted">No events yet.</td></tr>
                ) : audit.map((r) => (
                  <tr key={r.id}>
                    <td>{fmt(r.created_at)}</td><td>{r.actor}</td><td>{r.action}</td><td>{r.detail ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {tab === "about" && (
          <section className="panel">
            <h2>Architecture</h2>
            <p className="muted" style={{ lineHeight: 1.55 }}>
              React UI → FastAPI → PyTorch ResNet-18 → SQLite audit log.
              Trained on the public Brain Tumor MRI dataset (glioma, meningioma, pituitary, no tumor).
              CLAHE preprocessing and Grad-CAM explainability included.
            </p>
            <h2>Limitations</h2>
            <ul className="limitations">
              <li>Single 2D slice — not a replacement for volumetric MRI review.</li>
              <li>Requires validation on hospital scanners before clinical use.</li>
              <li>Grad-CAM shows model attention, not confirmed pathology.</li>
            </ul>
          </section>
        )}
      </main>
    </>
  );
}
