import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Card, CardContent, Typography, Grid, Box } from "@material-ui/core";
import Layout from "./Layout";
import { colors } from "../theme";

const useStyles = makeStyles((theme) => ({
  hero: {
    background: colors.gradient,
    borderRadius: 20,
    padding: theme.spacing(4),
    color: "#fff",
    marginBottom: theme.spacing(3),
    textAlign: "center",
  },
  card: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
    height: "100%",
  },
  step: {
    width: 36,
    height: 36,
    borderRadius: "50%",
    background: colors.gradient,
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 800,
    marginBottom: theme.spacing(1),
  },
}));

const steps = [
  { title: "Upload", desc: "Take a clear photo of a potato leaf and upload it." },
  { title: "Analyze", desc: "Our CNN model classifies Early Blight, Late Blight, or Healthy." },
  { title: "Get Advice", desc: "Gemini AI generates treatment and prevention recommendations." },
  { title: "Track", desc: "All scans are saved in your history dashboard." },
];

const tech = [
  { name: "TensorFlow", role: "Disease classification model" },
  { name: "FastAPI", role: "Python backend API" },
  { name: "React", role: "Web dashboard & UI" },
  { name: "Google Gemini", role: "AI treatment advice" },
  { name: "Google OAuth", role: "Secure sign-in" },
];

export default function About() {
  const classes = useStyles();

  return (
    <Layout title="About">
      <div className={classes.hero}>
        <Typography variant="h4" style={{ fontWeight: 800, marginBottom: 8 }}>
          About SpudGuard
        </Typography>
        <Typography style={{ opacity: 0.9, maxWidth: 560, margin: "0 auto" }}>
          An AI-powered platform helping farmers detect potato leaf diseases early and take action before crop loss.
        </Typography>
      </div>

      <Typography variant="h6" style={{ fontWeight: 700, marginBottom: 16, color: colors.text }}>
        How It Works
      </Typography>
      <Grid container spacing={2} style={{ marginBottom: 32 }}>
        {steps.map((s, i) => (
          <Grid item xs={12} sm={6} md={3} key={s.title}>
            <Card className={classes.card}>
              <CardContent>
                <div className={classes.step}>{i + 1}</div>
                <Typography style={{ fontWeight: 700, marginBottom: 4 }}>{s.title}</Typography>
                <Typography variant="body2" style={{ color: colors.textMuted }}>{s.desc}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" style={{ fontWeight: 700, marginBottom: 16, color: colors.text }}>
        Technology Stack
      </Typography>
      <Grid container spacing={2}>
        {tech.map((t) => (
          <Grid item xs={12} sm={6} md={4} key={t.name}>
            <Card className={classes.card}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography style={{ fontWeight: 700, color: colors.primary }}>{t.name}</Typography>
                </Box>
                <Typography variant="body2" style={{ color: colors.textMuted, marginTop: 4 }}>{t.role}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Layout>
  );
}
