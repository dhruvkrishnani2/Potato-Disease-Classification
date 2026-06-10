import React, { useState } from "react";
import { GoogleLogin } from "@react-oauth/google";
import { makeStyles } from "@material-ui/core/styles";
import {
  Paper,
  Typography,
  Container,
  Avatar,
  Grid,
  TextField,
  Button,
  Divider,
  Box,
  Tabs,
  Tab,
  InputAdornment,
  IconButton,
} from "@material-ui/core";
import EcoIcon from "@material-ui/icons/Eco";
import CameraAltIcon from "@material-ui/icons/CameraAlt";
import AndroidIcon from "@material-ui/icons/Android";
import SecurityIcon from "@material-ui/icons/Security";
import PersonIcon from "@material-ui/icons/Person";
import LockIcon from "@material-ui/icons/Lock";
import Visibility from "@material-ui/icons/Visibility";
import VisibilityOff from "@material-ui/icons/VisibilityOff";
import { useAuth } from "../context/AuthContext";
import { colors } from "../theme";

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || "";

const useStyles = makeStyles((theme) => ({
  root: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    background: `linear-gradient(135deg, ${colors.primaryDark} 0%, ${colors.primary} 40%, ${colors.primaryLight} 100%)`,
    position: "relative",
    overflow: "hidden",
    padding: theme.spacing(4, 0),
  },
  pattern: {
    position: "absolute",
    fontSize: 300,
    opacity: 0.04,
    right: -60,
    bottom: -60,
    userSelect: "none",
  },
  leftPanel: {
    color: "#fff",
    padding: theme.spacing(6),
    [theme.breakpoints.down("sm")]: { display: "none" },
  },
  heroTitle: { fontWeight: 800, fontSize: "2.5rem", lineHeight: 1.2, marginBottom: theme.spacing(2) },
  heroSub: { opacity: 0.85, fontSize: "1.05rem", lineHeight: 1.7, maxWidth: 400 },
  feature: {
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(2),
    marginTop: theme.spacing(3),
    opacity: 0.9,
  },
  featureIcon: {
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 12,
    padding: 10,
  },
  card: {
    padding: theme.spacing(4),
    borderRadius: 24,
    maxWidth: 440,
    width: "100%",
    boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
  },
  avatar: {
    width: 64,
    height: 64,
    margin: "0 auto 16px",
    background: colors.gradient,
  },
  title: { fontWeight: 800, color: colors.primary, fontSize: "1.6rem", textAlign: "center" },
  subtitle: {
    color: colors.textMuted,
    marginBottom: theme.spacing(2),
    lineHeight: 1.6,
    textAlign: "center",
    fontSize: "0.9rem",
  },
  tabs: {
    marginBottom: theme.spacing(2),
    "& .MuiTab-root": { textTransform: "none", fontWeight: 600, minWidth: 100 },
    "& .MuiTabs-indicator": { backgroundColor: colors.primaryLight },
  },
  field: { marginBottom: theme.spacing(2) },
  loginBtn: {
    borderRadius: 12,
    padding: "12px",
    fontWeight: 700,
    textTransform: "none",
    fontSize: "1rem",
    background: colors.gradient,
    color: "#fff",
    marginTop: theme.spacing(1),
    "&:hover": { background: colors.primary },
  },
  dividerRow: {
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(2),
    margin: theme.spacing(3, 0),
  },
  dividerLine: { flex: 1 },
  dividerText: {
    color: colors.textMuted,
    fontSize: "0.8rem",
    fontWeight: 600,
    whiteSpace: "nowrap",
  },
  googleWrap: { display: "flex", justifyContent: "center" },
  error: { color: colors.lateBlight, marginTop: theme.spacing(1.5), fontSize: "0.85rem", textAlign: "center" },
}));

const features = [
  { icon: <CameraAltIcon />, text: "Instant leaf disease detection" },
  { icon: <AndroidIcon />, text: "AI-powered treatment advice" },
  { icon: <SecurityIcon />, text: "Login with password or Google" },
];

export default function Login() {
  const classes = useStyles();
  const { loginWithGoogle, loginWithPassword, register } = useAuth();
  const [tab, setTab] = useState(0);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleGoogleSuccess = async (response) => {
    try {
      setError("");
      setSubmitting(true);
      await loginWithGoogle(response.credential);
    } catch (err) {
      setError("Google sign-in failed. Check that the API is running.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (tab === 0) {
        await loginWithPassword(username, password);
      } else {
        await register(username, password, name);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || "Authentication failed. Please try again.";
      setError(typeof msg === "string" ? msg : "Authentication failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={classes.root}>
      <span className={classes.pattern}>🥔</span>
      <Container maxWidth="lg">
        <Grid container alignItems="center" justifyContent="center" spacing={4}>
          <Grid item md={6} className={classes.leftPanel}>
            <Typography className={classes.heroTitle}>Protect Your Potato Crop with AI</Typography>
            <Typography className={classes.heroSub}>
              SpudGuard detects Early Blight, Late Blight, and healthy leaves — then gives you actionable treatment advice.
            </Typography>
            {features.map((f) => (
              <div key={f.text} className={classes.feature}>
                <div className={classes.featureIcon}>{f.icon}</div>
                <Typography>{f.text}</Typography>
              </div>
            ))}
          </Grid>

          <Grid item xs={12} md={5} style={{ display: "flex", justifyContent: "center" }}>
            <Paper className={classes.card} elevation={0}>
              <Avatar className={classes.avatar}>
                <EcoIcon style={{ fontSize: 32 }} />
              </Avatar>
              <Typography className={classes.title}>SpudGuard</Typography>
              <Typography className={classes.subtitle}>
                Sign in with username & password or continue with Google
              </Typography>

              <Tabs
                value={tab}
                onChange={(_, v) => { setTab(v); setError(""); }}
                variant="fullWidth"
                className={classes.tabs}
              >
                <Tab label="Sign In" />
                <Tab label="Register" />
              </Tabs>

              <form onSubmit={handleSubmit}>
                <TextField
                  className={classes.field}
                  fullWidth
                  label="Username"
                  variant="outlined"
                  size="small"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <PersonIcon style={{ color: colors.textMuted, fontSize: 20 }} />
                      </InputAdornment>
                    ),
                  }}
                />
                {tab === 1 && (
                  <TextField
                    className={classes.field}
                    fullWidth
                    label="Full Name (optional)"
                    variant="outlined"
                    size="small"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                )}
                <TextField
                  className={classes.field}
                  fullWidth
                  label="Password"
                  type={showPassword ? "text" : "password"}
                  variant="outlined"
                  size="small"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  helperText={tab === 1 ? "Minimum 6 characters" : ""}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <LockIcon style={{ color: colors.textMuted, fontSize: 20 }} />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton size="small" onClick={() => setShowPassword(!showPassword)}>
                          {showPassword ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  className={classes.loginBtn}
                  disabled={submitting}
                >
                  {submitting ? "Please wait..." : tab === 0 ? "Sign In" : "Create Account"}
                </Button>
              </form>

              {error && <Typography className={classes.error}>{error}</Typography>}

              {GOOGLE_CLIENT_ID && (
                <>
                  <div className={classes.dividerRow}>
                    <Divider className={classes.dividerLine} />
                    <span className={classes.dividerText}>OR</span>
                    <Divider className={classes.dividerLine} />
                  </div>
                  <Box className={classes.googleWrap}>
                    <GoogleLogin
                      onSuccess={handleGoogleSuccess}
                      onError={() => setError("Google sign-in was cancelled or failed.")}
                      theme="outline"
                      size="large"
                      text="continue_with"
                      shape="rectangular"
                    />
                  </Box>
                </>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}
