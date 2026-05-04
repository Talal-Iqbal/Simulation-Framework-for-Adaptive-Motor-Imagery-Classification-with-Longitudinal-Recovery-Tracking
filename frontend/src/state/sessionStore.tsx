import { createContext, useCallback, useContext, useEffect, useMemo, useReducer } from "react";
import type { RunState, SessionSnapshot, TrialPredictResponse } from "../types/api";

const STORAGE_PREFIX = "neurodrift.subject.";

interface SubjectRuntimeState {
  runState: RunState;
  sessionId: string;
  trials: TrialPredictResponse[];
  sessionSnapshot: SessionSnapshot | null;
  lastUpdatedAt: number;
}

interface AppState {
  currentSubjectId: number;
  subjects: Record<number, SubjectRuntimeState>;
}

type Action =
  | { type: "switchSubject"; subjectId: number }
  | { type: "setRunState"; runState: RunState }
  | { type: "appendTrial"; trial: TrialPredictResponse }
  | { type: "resetSession" }
  | { type: "setSnapshot"; snapshot: SessionSnapshot };

function nowSessionId() {
  return `sess-${Date.now()}`;
}

function buildSubjectState(subjectId: number): SubjectRuntimeState {
  const fromStorage = localStorage.getItem(`${STORAGE_PREFIX}${subjectId}`);
  if (fromStorage) {
    try {
      return JSON.parse(fromStorage) as SubjectRuntimeState;
    } catch {
      // Ignore broken storage and replace with a fresh state.
    }
  }
  return {
    runState: "idle",
    sessionId: nowSessionId(),
    trials: [],
    sessionSnapshot: null,
    lastUpdatedAt: Date.now()
  };
}

function snapshotsEqual(a: SessionSnapshot | null, b: SessionSnapshot): boolean {
  if (!a) return false;
  return (
    a.sessionId === b.sessionId &&
    a.subjectId === b.subjectId &&
    a.acceptedRate === b.acceptedRate &&
    a.gatedAccuracy === b.gatedAccuracy &&
    a.rejectRate === b.rejectRate &&
    a.frobDistToBaseline === b.frobDistToBaseline &&
    a.intertrialCovVar === b.intertrialCovVar &&
    a.estimatedR === b.estimatedR &&
    a.clusterId === b.clusterId &&
    a.assistanceState === b.assistanceState
  );
}

function reducer(state: AppState, action: Action): AppState {
  const subjectState = state.subjects[state.currentSubjectId] ?? buildSubjectState(state.currentSubjectId);
  switch (action.type) {
    case "switchSubject": {
      if (state.subjects[action.subjectId]) {
        return { ...state, currentSubjectId: action.subjectId };
      }
      return {
        currentSubjectId: action.subjectId,
        subjects: { ...state.subjects, [action.subjectId]: buildSubjectState(action.subjectId) }
      };
    }
    case "setRunState":
      if (subjectState.runState === action.runState) {
        return state;
      }
      return {
        ...state,
        subjects: {
          ...state.subjects,
          [state.currentSubjectId]: {
            ...subjectState,
            runState: action.runState,
            lastUpdatedAt: Date.now()
          }
        }
      };
    case "appendTrial":
      return {
        ...state,
        subjects: {
          ...state.subjects,
          [state.currentSubjectId]: {
            ...subjectState,
            trials: [...subjectState.trials, action.trial],
            lastUpdatedAt: Date.now()
          }
        }
      };
    case "setSnapshot":
      if (snapshotsEqual(subjectState.sessionSnapshot, action.snapshot)) {
        return state;
      }
      return {
        ...state,
        subjects: {
          ...state.subjects,
          [state.currentSubjectId]: {
            ...subjectState,
            sessionSnapshot: action.snapshot,
            lastUpdatedAt: Date.now()
          }
        }
      };
    case "resetSession":
      return {
        ...state,
        subjects: {
          ...state.subjects,
          [state.currentSubjectId]: {
            runState: "idle",
            sessionId: nowSessionId(),
            trials: [],
            sessionSnapshot: null,
            lastUpdatedAt: Date.now()
          }
        }
      };
    default:
      return state;
  }
}

interface SessionStoreValue {
  state: AppState;
  current: SubjectRuntimeState;
  switchSubject: (subjectId: number) => void;
  setRunState: (runState: RunState) => void;
  appendTrial: (trial: TrialPredictResponse) => void;
  resetSession: () => void;
  setSnapshot: (snapshot: SessionSnapshot) => void;
}

const SessionStoreContext = createContext<SessionStoreValue | null>(null);

export function SessionStoreProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => {
    const subjectIds = [1, 2, 3];
    const subjects = subjectIds.reduce<Record<number, SubjectRuntimeState>>((acc, id) => {
      acc[id] = buildSubjectState(id);
      return acc;
    }, {});
    return {
      currentSubjectId: 2,
      subjects
    };
  });

  const current = state.subjects[state.currentSubjectId];

  useEffect(() => {
    Object.entries(state.subjects).forEach(([subjectId, value]) => {
      localStorage.setItem(`${STORAGE_PREFIX}${subjectId}`, JSON.stringify(value));
    });
  }, [state.subjects]);

  const switchSubject = useCallback((subjectId: number) => {
    dispatch({ type: "switchSubject", subjectId });
  }, []);

  const setRunState = useCallback((runState: RunState) => {
    dispatch({ type: "setRunState", runState });
  }, []);

  const appendTrial = useCallback((trial: TrialPredictResponse) => {
    dispatch({ type: "appendTrial", trial });
  }, []);

  const resetSession = useCallback(() => {
    dispatch({ type: "resetSession" });
  }, []);

  const setSnapshot = useCallback((snapshot: SessionSnapshot) => {
    dispatch({ type: "setSnapshot", snapshot });
  }, []);

  const value: SessionStoreValue = useMemo(() => ({
    state,
    current,
    switchSubject,
    setRunState,
    appendTrial,
    resetSession,
    setSnapshot
  }), [appendTrial, current, resetSession, setRunState, setSnapshot, state, switchSubject]);

  return <SessionStoreContext.Provider value={value}>{children}</SessionStoreContext.Provider>;
}

export function useSessionStore() {
  const context = useContext(SessionStoreContext);
  if (!context) {
    throw new Error("useSessionStore must be used within SessionStoreProvider");
  }
  return context;
}
