import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Card, CardContent, Typography, Avatar, Grid, Chip, Box } from "@material-ui/core";
import EmailIcon from "@material-ui/icons/Email";
import PersonIcon from "@material-ui/icons/Person";
import VerifiedUserIcon from "@material-ui/icons/VerifiedUser";
import Layout from "./Layout";
import { useAuth } from "../context/AuthContext";
import { useHistory } from "../context/HistoryContext";
import { colors } from "../theme";

const useStyles = makeStyles((theme) => ({
  profileCard: {
    borderRadius: 20,
    boxShadow: colors.cardShadow,
    overflow: "visible",
  },
  banner: {
    background: colors.gradient,
    height: 120,
    borderRadius: "20px 20px 0 0",
    position: "relative",
  },
  avatar: {
    width: 96,
    height: 96,
    border: "4px solid #fff",
    position: "absolute",
    bottom: -48,
    left: theme.spacing(3),
    backgroundColor: colors.accent,
    fontSize: "2rem",
  },
  content: { paddingTop: theme.spacing(7) },
  infoRow: {
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(1.5),
    padding: theme.spacing(1.5, 0),
    borderBottom: `1px solid ${colors.background}`,
  },
  statCard: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
    textAlign: "center",
    padding: theme.spacing(2),
  },
  statValue: { fontWeight: 800, fontSize: "1.75rem", color: colors.primary },
  statLabel: { color: colors.textMuted, fontSize: "0.82rem" },
}));

export default function Profile() {
  const classes = useStyles();
  const { user } = useAuth();
  const { stats } = useHistory();
  const isGoogle = user?.sub && !user.sub.startsWith("local:");

  return (
    <Layout title="Profile">
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card className={classes.profileCard}>
            <div className={classes.banner}>
              <Avatar src={user?.picture} className={classes.avatar}>
                {user?.name?.[0]?.toUpperCase() || "?"}
              </Avatar>
            </div>
            <CardContent className={classes.content}>
              <Typography variant="h5" style={{ fontWeight: 800, color: colors.text }}>
                {user?.name}
              </Typography>
              <Chip
                icon={<VerifiedUserIcon />}
                label={isGoogle ? "Google Account" : "Email & Password"}
                size="small"
                style={{ marginTop: 8, backgroundColor: "#d8f3dc", color: colors.primary, fontWeight: 600 }}
              />

              <Box mt={3}>
                <div className={classes.infoRow}>
                  <PersonIcon style={{ color: colors.textMuted }} />
                  <div>
                    <Typography variant="caption" style={{ color: colors.textMuted }}>Display Name</Typography>
                    <Typography style={{ fontWeight: 600 }}>{user?.name}</Typography>
                  </div>
                </div>
                <div className={classes.infoRow}>
                  <EmailIcon style={{ color: colors.textMuted }} />
                  <div>
                    <Typography variant="caption" style={{ color: colors.textMuted }}>Email</Typography>
                    <Typography style={{ fontWeight: 600 }}>{user?.email}</Typography>
                  </div>
                </div>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" style={{ fontWeight: 700, marginBottom: 12, color: colors.text }}>
            Your Activity
          </Typography>
          <Grid container spacing={2}>
            {[
              { label: "Total Scans", value: stats.total },
              { label: "Healthy", value: stats.healthy },
              { label: "Diseases", value: stats.diseased },
            ].map((s) => (
              <Grid item xs={4} key={s.label}>
                <Card className={classes.statCard}>
                  <Typography className={classes.statValue}>{s.value}</Typography>
                  <Typography className={classes.statLabel}>{s.label}</Typography>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Layout>
  );
}
