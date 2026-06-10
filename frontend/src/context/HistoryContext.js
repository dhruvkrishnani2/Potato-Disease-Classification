import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import { useAuth } from "./AuthContext";

const HistoryContext = createContext(null);
const STORAGE_KEY = "spudguard_scan_history";

function loadHistory(email) {
  try {
    const raw = localStorage.getItem(`${STORAGE_KEY}_${email}`);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(email, history) {
  localStorage.setItem(`${STORAGE_KEY}_${email}`, JSON.stringify(history));
}

export function HistoryProvider({ children }) {
  const { user } = useAuth();
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (user?.email) {
      setHistory(loadHistory(user.email));
    } else {
      setHistory([]);
    }
  }, [user?.email]);

  const addScan = useCallback(
    (scan) => {
      if (!user?.email) return;
      const entry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        disease: scan.disease,
        confidence: scan.confidence,
        adviceSource: scan.adviceSource || null,
      };
      setHistory((prev) => {
        const updated = [entry, ...prev].slice(0, 50);
        saveHistory(user.email, updated);
        return updated;
      });
    },
    [user?.email]
  );

  const clearHistory = useCallback(() => {
    if (!user?.email) return;
    setHistory([]);
    saveHistory(user.email, []);
  }, [user?.email]);

  const stats = React.useMemo(() => {
    const total = history.length;
    const healthy = history.filter((h) => h.disease === "Healthy").length;
    const diseased = total - healthy;
    const avgConfidence =
      total > 0
        ? (history.reduce((sum, h) => sum + h.confidence, 0) / total) * 100
        : 0;
    const breakdown = {
      "Early Blight": history.filter((h) => h.disease === "Early Blight").length,
      "Late Blight": history.filter((h) => h.disease === "Late Blight").length,
      Healthy: healthy,
    };
    return { total, healthy, diseased, avgConfidence, breakdown };
  }, [history]);

  return (
    <HistoryContext.Provider value={{ history, addScan, clearHistory, stats }}>
      {children}
    </HistoryContext.Provider>
  );
}

export function useHistory() {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error("useHistory must be used within HistoryProvider");
  return ctx;
}
