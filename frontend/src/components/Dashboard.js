import React from "react";
import { useHistory } from "react-router-dom";
import { makeStyles } from "@material-ui/core/styles";
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
} from "@material-ui/core";
import CameraAltIcon from "@material-ui/icons/CameraAlt";
import CheckCircleIcon from "@material-ui/icons/CheckCircle";
import WarningIcon from "@material-ui/icons/Warning";
import AssessmentIcon from "@material-ui/icons/Assessment";
import TrendingUpIcon from "@material-ui/icons/TrendingUp";
import Layout from "./Layout";
import { useHistory as useScanHistory } from "../context/HistoryContext";
import { useAuth } from "../context/AuthContext";
import { colors, diseaseMeta, getDiseaseColor } from "../theme";

const useStyles = makeStyles((theme) => ({
  welcomeBanner: {
    background: colors.gradient,
    borderRadius: 20,
    padding: theme.spacing(4),
    color: "#fff",
    marginBottom: theme.spacing(3),
    position: "relative",
    overflow: "hidden",
    boxShadow: colors.cardShadow,
  },
  welcomePattern: {
    position: "absolute",
    right: -20,
    top: -20,
    fontSize: 140,
    opacity: 0.12,
    userSelect: "none",
  },
  welcomeTitle: { fontWeight: 800, fontSize: "1.75rem", marginBottom: theme.spacing(1) },
  welcomeSub: { opacity: 0.85, marginBottom: theme.spacing(2.5), maxWidth: 480 },
  ctaButton: {
    backgroundColor: "#fff",
    color: colors.primary,
    fontWeight: 700,
    borderRadius: 12,
    padding: "10px 28px",
    textTransform: "none",
    fontSize: "0.95rem",
    "&:hover": { backgroundColor: colors.accentLight },
  },
  statCard: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
    height: "100%",
    transition: "transform 0.2s, box-shadow 0.2s",
    "&:hover": { transform: "translateY(-2px)", boxShadow: colors.cardShadowHover },
  },
  statIcon: {
    width: 48,
    height: 48,
    borderRadius: 14,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: theme.spacing(1.5),
  },
  statValue: { fontWeight: 800, fontSize: "2rem", lineHeight: 1, color: colors.text },
  statLabel: { color: colors.textMuted, fontSize: "0.85rem", marginTop: 4 },
  sectionTitle: { fontWeight: 700, fontSize: "1.1rem", marginBottom: theme.spacing(2), color: colors.text },
  diseaseCard: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
    height: "100%",
    borderTop: "4px solid",
    transition: "transform 0.2s",
    "&:hover": { transform: "translateY(-2px)" },
  },
  diseaseIcon: { fontSize: "2rem", marginBottom: theme.spacing(1) },
  diseaseName: { fontWeight: 700, fontSize: "1rem" },
  diseaseDesc: { color: colors.textMuted, fontSize: "0.82rem", marginTop: 4 },
  recentCard: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
  },
  recentRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: theme.spacing(1.5, 0),
    borderBottom: `1px solid ${colors.background}`,
    "&:last-child": { borderBottom: "none" },
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    marginTop: theme.spacing(1),
    backgroundColor: colors.background,
  },
  emptyState: {
    textAlign: "center",
    padding: theme.spacing(4),
    color: colors.textMuted,
  },
}));

function StatCard({ icon, label, value, suffix, color, bgColor }) {
  const classes = useStyles();
  return (
    <Card className={classes.statCard}>
      <CardContent>
        <div className={classes.statIcon} style={{ backgroundColor: bgColor, color }}>
          {icon}
        </div>
        <Typography className={classes.statValue}>
          {value}
          {suffix && (
            <span style={{ fontSize: "1rem", fontWeight: 600, color: colors.textMuted }}>{suffix}</span>
          )}
        </Typography>
        <Typography className={classes.statLabel}>{label}</Typography>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const classes = useStyles();
  const history = useHistory();
  const { user } = useAuth();
  const { history: scans, stats } = useScanHistory();
  const firstName = user?.name?.split(" ")[0] || "Farmer";

  return (
    <Layout title="Dashboard">
      <div className={classes.welcomeBanner}>
        <span className={classes.welcomePattern}>🥔</span>
        <Typography className={classes.welcomeTitle}>Welcome back, {firstName}!</Typography>
        <Typography className={classes.welcomeSub}>
          Monitor your potato crop health with AI-powered disease detection and personalized treatment advice.
        </Typography>
        <Button
          className={classes.ctaButton}
          startIcon={<CameraAltIcon />}
          onClick={() => history.push("/classify")}
        >
          Scan a New Leaf
        </Button>
      </div>

      <Grid container spacing={3} style={{ marginBottom: 24 }}>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<AssessmentIcon />}
            label="Total Scans"
            value={stats.total}
            color={colors.primary}
            bgColor="#d8f3dc"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<CheckCircleIcon />}
            label="Healthy Plants"
            value={stats.healthy}
            color={colors.healthy}
            bgColor="#d8f3dc"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<WarningIcon />}
            label="Diseases Found"
            value={stats.diseased}
            color={colors.lateBlight}
            bgColor="#fde8e8"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<TrendingUpIcon />}
            label="Avg Confidence"
            value={stats.total > 0 ? stats.avgConfidence.toFixed(1) : "—"}
            suffix={stats.total > 0 ? "%" : ""}
            color={colors.accent}
            bgColor="#e8f5e9"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Typography className={classes.sectionTitle}>Recent Scans</Typography>
          <Card className={classes.recentCard}>
            <CardContent>
              {scans.length === 0 ? (
                <div className={classes.emptyState}>
                  <Typography variant="h3" style={{ marginBottom: 8 }}>
                    📷
                  </Typography>
                  <Typography>No scans yet</Typography>
                  <Typography variant="body2" style={{ marginTop: 4 }}>
                    Upload a potato leaf image to get started
                  </Typography>
                  <Button
                    className={classes.ctaButton}
                    style={{ marginTop: 16, background: colors.gradient, color: "#fff" }}
                    onClick={() => history.push("/classify")}
                  >
                    Start Scanning
                  </Button>
                </div>
              ) : (
                scans.slice(0, 6).map((scan) => (
                  <div key={scan.id} className={classes.recentRow}>
                    <Box display="flex" alignItems="center" style={{ gap: 12 }}>
                      <span style={{ fontSize: "1.5rem" }}>{diseaseMeta[scan.disease]?.icon || "🌱"}</span>
                      <div>
                        <Typography style={{ fontWeight: 600, fontSize: "0.9rem" }}>{scan.disease}</Typography>
                        <Typography variant="caption" style={{ color: colors.textMuted }}>
                          {new Date(scan.timestamp).toLocaleString()}
                        </Typography>
                      </div>
                    </Box>
                    <Chip
                      label={`${(scan.confidence * 100).toFixed(1)}%`}
                      size="small"
                      style={{
                        backgroundColor: `${getDiseaseColor(scan.disease)}18`,
                        color: getDiseaseColor(scan.disease),
                        fontWeight: 700,
                      }}
                    />
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Typography className={classes.sectionTitle}>Disease Distribution</Typography>
          <Card className={classes.recentCard}>
            <CardContent>
              {stats.total === 0 ? (
                <Typography variant="body2" style={{ color: colors.textMuted, textAlign: "center", padding: 24 }}>
                  Distribution will appear after your first scan
                </Typography>
              ) : (
                Object.entries(stats.breakdown).map(([disease, count]) => {
                  const pct = stats.total > 0 ? (count / stats.total) * 100 : 0;
                  const meta = diseaseMeta[disease];
                  return (
                    <Box key={disease} mb={2.5}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography style={{ fontWeight: 600, fontSize: "0.85rem" }}>
                          {meta?.icon} {disease}
                        </Typography>
                        <Typography variant="caption" style={{ color: colors.textMuted }}>
                          {count} ({pct.toFixed(0)}%)
                        </Typography>
                      </Box>
                      <Box className={classes.progressBar}>
                        <Box
                          style={{
                            height: "100%",
                            width: `${pct}%`,
                            backgroundColor: meta?.color,
                            borderRadius: 4,
                            transition: "width 0.6s ease",
                          }}
                        />
                      </Box>
                    </Box>
                  );
                })
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Typography className={classes.sectionTitle}>Know Your Diseases</Typography>
          <Grid container spacing={2}>
            {Object.entries(diseaseMeta).map(([name, meta]) => (
              <Grid item xs={12} sm={4} key={name}>
                <Card className={classes.diseaseCard} style={{ borderTopColor: meta.color }}>
                  <CardContent>
                    <div className={classes.diseaseIcon}>{meta.icon}</div>
                    <Typography className={classes.diseaseName} style={{ color: meta.color }}>
                      {name}
                    </Typography>
                    <Typography className={classes.diseaseDesc}>{meta.description}</Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Layout>
  );
}
