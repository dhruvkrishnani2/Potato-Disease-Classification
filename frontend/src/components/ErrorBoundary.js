import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: "Inter, sans-serif", maxWidth: 560, margin: "40px auto" }}>
          <h2 style={{ color: "#c1121f" }}>Something went wrong</h2>
          <p style={{ color: "#555", lineHeight: 1.6 }}>
            The app crashed while loading. Try clearing site data and refreshing.
          </p>
          <pre
            style={{
              background: "#f5f5f5",
              padding: 16,
              borderRadius: 8,
              overflow: "auto",
              fontSize: 13,
            }}
          >
            {this.state.error.message}
          </pre>
          <button
            type="button"
            onClick={() => {
              localStorage.removeItem("spudguard_token");
              window.location.href = "/";
            }}
            style={{
              marginTop: 16,
              padding: "10px 20px",
              background: "#2d6a4f",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Clear session & reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
