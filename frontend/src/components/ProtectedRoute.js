import React from "react";
import { CircularProgress, Box } from "@material-ui/core";
import { useAuth } from "../context/AuthContext";
import Login from "./Login";

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        style={{ background: "#f0f7f4" }}
      >
        <CircularProgress style={{ color: "#2d6a4f" }} />
        <p style={{ marginTop: 16, color: "#2d6a4f", fontWeight: 600 }}>Loading SpudGuard...</p>
      </Box>
    );
  }

  if (!user) {
    return <Login />;
  }

  return children;
}
