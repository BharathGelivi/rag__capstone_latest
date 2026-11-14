import { useEffect, useState } from "react";
import { API_BASE, getDiagnosticReports, type DiagnosticReport } from "../lib/api";

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function Reports() {
  const [reports, setReports] = useState<DiagnosticReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getDiagnosticReports().then(setReports).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, []);

  return <div className="reports-page">
    <div className="topbar"><div><span className="topbar-title">Diagnostic reports</span><p className="text-muted">Saved evidence, findings, and corrective actions.</p></div></div>
    <main className="reports-content">
      {!loading && !error && reports.length > 0 && <div className="reports-summary"><strong>{reports.length}</strong><span>generated diagnostic reports</span></div>}
      {loading && <p className="text-muted">Loading reports…</p>}
      {error && <p className="text-muted">Could not load reports: {error}</p>}
      {!loading && !error && reports.length === 0 && <p className="text-muted">No diagnostic reports have been generated yet.</p>}
      <div className="reports-list">{reports.map((report) => <article className="card report-row" key={report.name}>
        <div><div className="card-title">{report.trace_id}</div><div className="text-muted">{new Date(report.modified_at * 1000).toLocaleString()} · {formatBytes(report.size_bytes)}</div></div>
        <a className="btn btn-secondary" href={`${API_BASE}${report.url}`} target="_blank" rel="noreferrer">Open report</a>
      </article>)}</div>
    </main>
  </div>;
}
