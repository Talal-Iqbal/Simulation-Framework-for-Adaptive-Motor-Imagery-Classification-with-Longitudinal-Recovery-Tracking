import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { API_BASE, ApiError, analyzeSession, getHealth, getNextEvalTrial, startEvalSession } from "./lib/apiClient";
import { buildLayer6Series, mockAnalyzeSession, mockSessionFeatures, mockTrial } from "./lib/mockData";
import { useSessionStore } from "./state/sessionStore";
import type { NavKey, SessionFeatureVector } from "./types/api";

const NAV_ITEMS: { key: NavKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "session", label: "Session Run" },
  { key: "calibration", label: "Calibration" },
  { key: "layer6", label: "Layer 6" },
  { key: "layer7", label: "Layer 7" },
  { key: "diagnostics", label: "Diagnostics" },
  { key: "subjects", label: "Subject Manager" }
];

const CLASSIFICATION_MS = 50;
const FEEDBACK_MS = 200;
const ITI_MS = 1500;

function metricCard(label: string, value: string, helper?: string) {
  return (
    <article className="metric-card" role="status" aria-label={label}>
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {helper ? <p className="metric-helper">{helper}</p> : null}
    </article>
  );
}

function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function statusTone(status: "normal" | "watch" | "assist") {
  if (status === "assist") return "state-danger";
  if (status === "watch") return "state-warn";
  return "state-ok";
}

export default function App() {
  const { state, current, switchSubject, setRunState, appendTrial, resetSession, setSnapshot, setEvalSessionId } =
    useSessionStore();

  const [activeNav, setActiveNav] = useState<NavKey>("overview");
  const [shape, setShape] = useState<"linear" | "plateau" | "relapse">("linear");
  const [healthStatus, setHealthStatus] = useState<string>("checking");
  const [healthInfo, setHealthInfo] = useState<string>("Connecting to API...");
  const [lastError, setLastError] = useState<string | null>(null);
  const [sessionFeatures, setSessionFeatures] = useState<SessionFeatureVector>(mockSessionFeatures());
  const [layer7Loading, setLayer7Loading] = useState(false);

  const trialSummary = useMemo(() => {
    const total = current.trials.length;
    const accepted = current.trials.filter((trial) => trial.accepted).length;
    const rejected = total - accepted;
    const correct = current.trials.filter((trial) => trial.correct).length;
    const acceptedCorrect = current.trials.filter((trial) => trial.accepted && trial.correct).length;

    return {
      total,
      acceptedRate: total ? accepted / total : 0,
      rejectRate: total ? rejected / total : 0,
      accuracyAll: total ? correct / total : 0,
      accuracyAccepted: accepted ? acceptedCorrect / accepted : 0
    };
  }, [current.trials]);

  const confidenceSeries = current.trials.map((trial) => ({
    idx: trial.trial_idx,
    confidence: Number(trial.confidence.toFixed(3)),
    margin: Number(trial.margin.toFixed(3))
  }));

  const rejectionSeries = useMemo(() => {
    const counts: Record<string, number> = {};
    current.trials.forEach((trial) => {
      if (trial.accepted) return;
      trial.reject_reasons.forEach((reason) => {
        const base = reason.split("(")[0];
        counts[base] = (counts[base] ?? 0) + 1;
      });
    });
    return Object.entries(counts).map(([reason, count]) => ({ reason, count }));
  }, [current.trials]);

  const layer6Series = useMemo(() => buildLayer6Series(shape), [shape]);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((health) => {
        if (cancelled) return;
        setHealthStatus("online");
        const hasCalibration = Boolean(health.models[`calibration_subject_${state.currentSubjectId}`]);
        setHealthInfo(
          hasCalibration
            ? `API online (v${health.version}) • subject calibration available`
            : `API online (v${health.version}) • calibration missing for subject ${state.currentSubjectId}`
        );
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setHealthStatus("fallback");
        setHealthInfo("Mock mode active (API unavailable or blocked)");
        if (error instanceof Error) {
          setLastError(error.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [state.currentSubjectId]);

  useEffect(() => {
    if (current.runState !== "running") return;
    if (!current.evalSessionId) return;

    const evalSessionId = current.evalSessionId;

    const timer = window.setInterval(async () => {
      const lastTrial = current.trials[current.trials.length - 1];
      try {
        const trial = await getNextEvalTrial(evalSessionId);
        appendTrial(trial);
        if (trial.exhausted) {
          setRunState("ended");
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 410) {
          setRunState("ended");
          return;
        }
        const fallback = mockTrial(
          (lastTrial?.trial_idx ?? 0) + 1,
          lastTrial?.timestamp_s ?? 0
        );
        appendTrial(fallback);
        if (error instanceof ApiError) {
          setLastError(`${error.message} (falling back to mock predictions)`);
        }
      }
    }, 1200);

    return () => window.clearInterval(timer);
  }, [appendTrial, current.evalSessionId, current.runState, current.trials, setRunState]);

  useEffect(() => {
    if (!current.trials.length) return;
    const assistanceState =
      trialSummary.rejectRate > 0.4 ? "assist" : trialSummary.rejectRate > 0.2 ? "watch" : "normal";
    const estimatedR = current.sessionSnapshot?.estimatedR ?? 0.65;
    const clusterId = current.sessionSnapshot?.clusterId ?? 1;

    setSnapshot({
      sessionId: current.sessionId,
      subjectId: state.currentSubjectId,
      acceptedRate: trialSummary.acceptedRate,
      gatedAccuracy: trialSummary.accuracyAccepted,
      rejectRate: trialSummary.rejectRate,
      frobDistToBaseline: sessionFeatures.frob_dist_to_baseline,
      intertrialCovVar: sessionFeatures.intertrial_cov_var,
      estimatedR,
      clusterId,
      assistanceState
    });
  }, [
    current.sessionId,
    current.trials.length,
    current.sessionSnapshot?.clusterId,
    current.sessionSnapshot?.estimatedR,
    sessionFeatures.frob_dist_to_baseline,
    sessionFeatures.intertrial_cov_var,
    setSnapshot,
    state.currentSubjectId,
    trialSummary.acceptedRate,
    trialSummary.accuracyAccepted,
    trialSummary.rejectRate
  ]);

  async function refreshLayer7Analysis() {
    setLayer7Loading(true);
    try {
      const result = await analyzeSession(sessionFeatures);
      const assistanceState =
        result.predicted_r < 0.4 ? "assist" : result.predicted_r < 0.62 ? "watch" : "normal";

      setSnapshot({
        sessionId: current.sessionId,
        subjectId: state.currentSubjectId,
        acceptedRate: trialSummary.acceptedRate,
        gatedAccuracy: trialSummary.accuracyAccepted,
        rejectRate: trialSummary.rejectRate,
        frobDistToBaseline: sessionFeatures.frob_dist_to_baseline,
        intertrialCovVar: sessionFeatures.intertrial_cov_var,
        estimatedR: result.predicted_r,
        clusterId: result.cluster,
        assistanceState
      });
    } catch (error) {
      const result = mockAnalyzeSession(sessionFeatures);
      const assistanceState =
        result.predicted_r < 0.4 ? "assist" : result.predicted_r < 0.62 ? "watch" : "normal";

      setSnapshot({
        sessionId: current.sessionId,
        subjectId: state.currentSubjectId,
        acceptedRate: trialSummary.acceptedRate,
        gatedAccuracy: trialSummary.accuracyAccepted,
        rejectRate: trialSummary.rejectRate,
        frobDistToBaseline: sessionFeatures.frob_dist_to_baseline,
        intertrialCovVar: sessionFeatures.intertrial_cov_var,
        estimatedR: result.predicted_r,
        clusterId: result.cluster,
        assistanceState
      });
      setLastError(error instanceof Error ? `${error.message} (using Layer 7 mock)` : "Layer 7 fallback");
    } finally {
      setLayer7Loading(false);
    }
  }

  async function onStart() {
    setLastError(null);
    try {
      const session = await startEvalSession(state.currentSubjectId);
      setEvalSessionId(session.session_id);
      setRunState("running");
    } catch (error) {
      const msg = error instanceof ApiError ? error.message : "Failed to start eval session";
      setLastError(`${msg} — check that the subject is calibrated and the API is online`);
    }
  }

  function onPause() {
    if (current.runState !== "running") return;
    setRunState("paused");
  }

  function onResume() {
    if (current.runState !== "paused") return;
    setRunState("running");
  }

  function onEnd() {
    setRunState("ended");
  }

  return (
    <div className="app-shell">
      <aside className="side-nav" aria-label="Primary">
        <h1>NeuroDrift</h1>
        <p className="nav-subtitle">EEG session intelligence</p>
        <nav>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              className={activeNav === item.key ? "nav-btn active" : "nav-btn"}
              onClick={() => setActiveNav(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <p className="api-base" title={API_BASE}>
          API: {API_BASE}
        </p>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="subject-switch">
            <label htmlFor="subject">Subject</label>
            <select
              id="subject"
              value={state.currentSubjectId}
              onChange={(event) => switchSubject(Number(event.target.value))}
            >
              {Object.keys(state.subjects).map((subjectId) => (
                <option key={subjectId} value={subjectId}>
                  Subject {subjectId}
                </option>
              ))}
            </select>
          </div>

          <div className="session-controls" role="group" aria-label="Session controls">
            <button onClick={onStart} disabled={current.runState === "running"}>
              Start
            </button>
            <button onClick={onPause} disabled={current.runState !== "running"}>
              Pause
            </button>
            <button onClick={onResume} disabled={current.runState !== "paused"}>
              Resume
            </button>
            <button onClick={onEnd} disabled={current.runState === "ended" || current.runState === "idle"}>
              End
            </button>
            <button onClick={resetSession}>Reset</button>
          </div>

          <div className="status-strip">
            <span className={`status-pill status-${current.runState}`}>{current.runState}</span>
            <span className={`status-pill status-${healthStatus}`}>{healthInfo}</span>
          </div>
        </header>

        {lastError ? (
          <section className="alert" aria-live="polite">
            {lastError}
          </section>
        ) : null}

        <section className="kpi-grid">
          {metricCard("Trials", String(trialSummary.total), "Per-subject isolated state")}
          {metricCard("Accepted", formatPct(trialSummary.acceptedRate))}
          {metricCard("Rejected", formatPct(trialSummary.rejectRate))}
          {metricCard("Gated Accuracy", formatPct(trialSummary.accuracyAccepted))}
          {metricCard(
            "Estimated r",
            current.sessionSnapshot ? current.sessionSnapshot.estimatedR.toFixed(3) : "n/a"
          )}
          {metricCard(
            "Assistance",
            current.sessionSnapshot?.assistanceState ?? "normal",
            "Derived from Layer 7 / reject rate"
          )}
        </section>

        <section className="timing-strip" aria-label="Signal timing timeline">
          <h2>Trial Timing (Simulated where unavailable)</h2>
          <div className="timing-bars">
            <div style={{ flex: CLASSIFICATION_MS }} className="timing-segment classification">
              Classification {CLASSIFICATION_MS}ms
            </div>
            <div style={{ flex: FEEDBACK_MS }} className="timing-segment feedback">
              Feedback {FEEDBACK_MS}ms
            </div>
            <div style={{ flex: ITI_MS }} className="timing-segment iti">
              ITI {ITI_MS}ms
            </div>
          </div>
          <p className="timing-note">
            Latency badge:{" "}
            <strong>{healthStatus === "online" ? "on-time (classification from API)" : "unknown (mock mode)"}</strong>
          </p>
        </section>

        {activeNav === "overview" ? (
          <section className="panel-grid">
            <article className="panel">
              <h3>Confidence / Margin Trend</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={confidenceSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" />
                  <XAxis dataKey="idx" tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <YAxis domain={[-1, 1]} tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="confidence" stroke="#00d4b8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="margin" stroke="#a78bfa" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </article>
            <article className="panel">
              <h3>Rejection Breakdown</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rejectionSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" />
                  <XAxis dataKey="reason" tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <YAxis tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f87171" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </article>
          </section>
        ) : null}

        {activeNav === "session" ? (
          <section className="panel">
            <h3>Per-trial Event Stream</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Trial</th>
                    <th>Accepted</th>
                    <th>Prediction</th>
                    <th>Truth</th>
                    <th>Confidence</th>
                    <th>Margin</th>
                    <th>Correct</th>
                    <th>Reject Reasons</th>
                    <th>Timestamp (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {current.trials.slice().reverse().slice(0, 120).map((trial, idx) => (
                    <tr key={`${trial.trial_idx}-${trial.timestamp_s}-${idx}`}>
                      <td>{trial.trial_idx}</td>
                      <td>{trial.accepted ? "Yes" : "No"}</td>
                      <td>{trial.y_pred}</td>
                      <td>{trial.y_true}</td>
                      <td>{trial.confidence.toFixed(3)}</td>
                      <td>{trial.margin.toFixed(3)}</td>
                      <td>{trial.correct ? "Yes" : "No"}</td>
                      <td>{trial.reject_reasons.join(", ") || "-"}</td>
                      <td>{trial.timestamp_s.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {activeNav === "calibration" ? (
          <section className="panel-grid">
            <article className="panel">
              <h3>Calibration Readiness</h3>
              <ul className="checklist">
                <li>Accepted rate target: {trialSummary.acceptedRate > 0.65 ? "pass" : "watch"}</li>
                <li>Confidence stability: {trialSummary.accuracyAll > 0.6 ? "pass" : "watch"}</li>
                <li>Reject pressure: {trialSummary.rejectRate < 0.3 ? "pass" : "fail"}</li>
                <li>Recommendation: {trialSummary.rejectRate > 0.35 ? "continue calibration" : "ready to deploy"}</li>
              </ul>
            </article>
            <article className="panel">
              <h3>Session Health Snapshot</h3>
              {current.sessionSnapshot ? (
                <dl className="snapshot">
                  <dt>Session ID</dt>
                  <dd>{current.sessionSnapshot.sessionId}</dd>
                  <dt>Frobenius Drift</dt>
                  <dd>{current.sessionSnapshot.frobDistToBaseline.toFixed(3)}</dd>
                  <dt>Intertrial Covariance Variance</dt>
                  <dd>{current.sessionSnapshot.intertrialCovVar.toFixed(3)}</dd>
                  <dt>Assistance State</dt>
                  <dd className={statusTone(current.sessionSnapshot.assistanceState)}>
                    {current.sessionSnapshot.assistanceState}
                  </dd>
                </dl>
              ) : (
                <p>Start a session to generate calibration metrics.</p>
              )}
            </article>
          </section>
        ) : null}

        {activeNav === "layer6" ? (
          <section className="panel-grid">
            <article className="panel">
              <h3>Trajectory Controls</h3>
              <div className="control-row">
                <label htmlFor="shape">Shape</label>
                <select
                  id="shape"
                  value={shape}
                  onChange={(event) => setShape(event.target.value as "linear" | "plateau" | "relapse")}
                >
                  <option value="linear">Linear</option>
                  <option value="plateau">Plateau</option>
                  <option value="relapse">Relapse</option>
                </select>
              </div>
              <p>
                Layer 6 simulates degradation/recovery profile and its impact on classifier behavior and ERD drift.
              </p>
            </article>
            <article className="panel">
              <h3>Layer 6 Output</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={layer6Series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" />
                  <XAxis dataKey="session" tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fill: "#7a8aaa", fontSize: 11 }} axisLine={{ stroke: "#1e2d47" }} tickLine={false} />
                  <Tooltip />
                  <Legend />
                  <Line dataKey="r" stroke="#00d4b8" strokeWidth={2} dot={false} />
                  <Line dataKey="ldaAcc" stroke="#34d399" strokeWidth={2} dot={false} />
                  <Line dataKey="erdDrift" stroke="#fbbf24" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </article>
          </section>
        ) : null}

        {activeNav === "layer7" ? (
          <section className="panel-grid">
            <article className="panel">
              <h3>Session Features (Input)</h3>
              <div className="feature-actions">
                <button onClick={() => setSessionFeatures(mockSessionFeatures())}>Regenerate features</button>
                <button onClick={refreshLayer7Analysis} disabled={layer7Loading}>
                  {layer7Loading ? "Analyzing..." : "Analyze Session"}
                </button>
              </div>
              <div className="feature-grid">
                {Object.entries(sessionFeatures).map(([key, value]) => (
                  <div className="feature-item" key={key}>
                    <span>{key}</span>
                    <strong>{value.toFixed(3)}</strong>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel">
              <h3>Layer 7 Inference (Processing/Output)</h3>
              {current.sessionSnapshot ? (
                <dl className="snapshot">
                  <dt>Predicted r</dt>
                  <dd>{current.sessionSnapshot.estimatedR.toFixed(3)}</dd>
                  <dt>Cluster ID</dt>
                  <dd>{current.sessionSnapshot.clusterId}</dd>
                  <dt>Assistance</dt>
                  <dd className={statusTone(current.sessionSnapshot.assistanceState)}>
                    {current.sessionSnapshot.assistanceState}
                  </dd>
                </dl>
              ) : (
                <p>Run Layer 7 analysis to produce cluster and recovery estimates.</p>
              )}
            </article>
          </section>
        ) : null}

        {activeNav === "diagnostics" ? (
          <section className="panel">
            <h3>Backend Diagnostics</h3>
            <ul className="checklist">
              <li>Transport: REST (no WebSocket/SSE currently)</li>
              <li>Real-time strategy: polling + event buffering</li>
              <li>Pipeline readiness: inferred from /health and API status codes</li>
              <li>Fallback mode: mock data for uninterrupted UX testing</li>
            </ul>
          </section>
        ) : null}

        {activeNav === "subjects" ? (
          <section className="panel">
            <h3>Subject State Isolation</h3>
            <p>
              Each subject keeps a dedicated session state in localStorage, including trial log, session snapshot, and
              run status. Switching subjects never overwrites another subject&apos;s active session.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Subject</th>
                    <th>Run state</th>
                    <th>Trials</th>
                    <th>Last update</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(state.subjects).map(([subjectId, subjectState]) => (
                    <tr key={subjectId}>
                      <td>{subjectId}</td>
                      <td>{subjectState.runState}</td>
                      <td>{subjectState.trials.length}</td>
                      <td>{new Date(subjectState.lastUpdatedAt).toLocaleTimeString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
