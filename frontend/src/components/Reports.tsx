import { useEffect, useState } from "react";
import { API_BASE, getDiagnosticReports, type DiagnosticReport, type Session } from "../lib/api";

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function Reports({ sessions, onOpenSession }: {
  sessions: Session[];
  onOpenSession: (sessionId: string) => void;
}) {
  const [reports, setReports] = useState<DiagnosticReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getDiagnosticReports().then(setReports).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, []);

  return <div className="reports-page">
    <div className="topbar"><div><span className="topbar-title">Report artifacts</span><p className="text-muted">Saved chat sessions and diagnostic reports.</p></div></div>
    <main className="reports-content">
      {loading && <p className="text-muted">Loading reports…</p>}
      {error && <p className="text-muted">Could not load reports: {error}</p>}
      {!loading && !error && reports.length === 0 && <p className="text-muted">No reports have been saved yet.</p>}
      <div className="reports-list">{reports.map((report) => {
        const session = sessions.find((entry) => entry.session_id === report.session_id);
        return <article className="card report-row" key={report.name}>
          <div>
            <div className="card-title">{report.session_id ? session?.title || report.session_id : report.trace_id}</div>
            <div className="text-muted">{report.session_id ? "Session report" : "Diagnostic report"} · {new Date(report.modified_at * 1000).toLocaleString()} · {formatBytes(report.size_bytes)}</div>
          </div>
          <div className="report-actions">
            {session && <button className="btn btn-secondary" onClick={() => onOpenSession(session.session_id)}>Continue chat</button>}
            <a className="btn btn-secondary" href={`${API_BASE}${report.url}`} target="_blank" rel="noreferrer">Open report</a>
          </div>
        </article>;
      })}</div>
    </main>
  </div>;
}
