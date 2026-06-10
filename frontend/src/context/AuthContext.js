import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import axios from "axios";

const AuthContext = createContext(null);

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const API_TIMEOUT = 8000;

const api = axios.create({ baseURL: API_BASE, timeout: API_TIMEOUT });

function saveSession(access_token, userData, setToken, setUser) {
  localStorage.setItem("spudguard_token", access_token);
  setToken(access_token);
  setUser(userData);
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("spudguard_token"));
  const [loading, setLoading] = useState(true);

  const authHeaders = useCallback(
    () => (token ? { Authorization: `Bearer ${token}` } : {}),
    [token]
  );

  const logout = useCallback(() => {
    localStorage.removeItem("spudguard_token");
    setToken(null);
    setUser(null);
  }, []);

  const loginWithGoogle = useCallback(async (credential) => {
    const res = await api.post("/auth/google", { credential });
    const { access_token, user: userData } = res.data;
    saveSession(access_token, userData, setToken, setUser);
    return userData;
  }, []);

  const loginWithPassword = useCallback(async (username, password) => {
    const res = await api.post("/auth/login", { username, password });
    const { access_token, user: userData } = res.data;
    saveSession(access_token, userData, setToken, setUser);
    return userData;
  }, []);

  const register = useCallback(async (username, password, name = "", email = "") => {
    const res = await api.post("/auth/register", { username, password, name, email });
    const { access_token, user: userData } = res.data;
    saveSession(access_token, userData, setToken, setUser);
    return userData;
  }, []);

  useEffect(() => {
    if (!token) {
      setUser(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    api
      .get("/auth/me", { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => {
        if (!cancelled) setUser(res.data);
      })
      .catch(() => {
        if (!cancelled) logout();
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token, logout]);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        loginWithGoogle,
        loginWithPassword,
        register,
        logout,
        authHeaders,
        API_BASE,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
