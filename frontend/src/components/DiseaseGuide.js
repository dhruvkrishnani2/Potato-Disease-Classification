import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Grid, Card, CardContent, Typography, Chip, Box } from "@material-ui/core";
import Layout from "./Layout";
import { colors, diseaseMeta } from "../theme";

const useStyles = makeStyles((theme) => ({
  intro: {
    background: colors.gradient,
    borderRadius: 20,
    padding: theme.spacing(4),
    color: "#fff",
    marginBottom: theme.spacing(3),
  },
  card: {
    borderRadius: 16,
    boxShadow: colors.cardShadow,
    height: "100%",
    borderTop: "4px solid",
  },
  icon: { fontSize: "2.5rem", marginBottom: theme.spacing(1) },
  section: { marginTop: theme.spacing(1.5) },
  sectionTitle: { fontWeight: 700, fontSize: "0.85rem", color: colors.primary, marginBottom: 4 },
  sectionText: { fontSize: "0.88rem", color: colors.textMuted, lineHeight: 1.6 },
}));

const details = {
  "Early Blight": {
    symptoms: "Dark brown spots with target-like rings on older leaves. Yellowing spreads outward from lesions.",
    treatment: "Remove infected foliage, apply chlorothalonil or copper fungicide, improve spacing for airflow.",
    prevention: "Crop rotation (3+ years), certified seed, avoid overhead irrigation, mulch around plants.",
  },
  "Late Blight": {
    symptoms: "Water-soaked gray-green patches turning brown-black. White fuzzy mold on leaf undersides in humidity.",
    treatment: "Destroy infected plants immediately. Apply mancozeb or metalaxyl. Act within 24 hours.",
    prevention: "Plant resistant varieties, scout daily in wet weather, destroy volunteer potatoes and cull piles.",
  },
  Healthy: {
    symptoms: "Uniform green color, no spots or wilting. Firm leaf texture with clear veins.",
    treatment: "No treatment needed. Continue balanced NPK fertilization and regular watering.",
    prevention: "Weekly scouting, sanitize tools, maintain 2-week fungicide schedule as preventive in high-risk areas.",
  },
};

export default function DiseaseGuide() {
  const classes = useStyles();

  return (
    <Layout title="Disease Guide">
      <div className={classes.intro}>
        <Typography variant="h5" style={{ fontWeight: 800, marginBottom: 8 }}>
          Potato Disease Reference
        </Typography>
        <Typography style={{ opacity: 0.9 }}>
          Learn to identify the three conditions SpudGuard detects and how to manage each one.
        </Typography>
      </div>

      <Grid container spacing={3}>
        {Object.entries(diseaseMeta).map(([name, meta]) => (
          <Grid item xs={12} md={4} key={name}>
            <Card className={classes.card} style={{ borderTopColor: meta.color }}>
              <CardContent>
                <div className={classes.icon}>{meta.icon}</div>
                <Typography variant="h6" style={{ fontWeight: 700, color: meta.color }}>
                  {name}
                </Typography>
                <Typography variant="body2" style={{ color: colors.textMuted, marginBottom: 12 }}>
                  {meta.description}
                </Typography>
                {Object.entries(details[name]).map(([key, text]) => (
                  <Box key={key} className={classes.section}>
                    <Typography className={classes.sectionTitle}>
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </Typography>
                    <Typography className={classes.sectionText}>{text}</Typography>
                  </Box>
                ))}
                <Chip
                  label="Detectable by SpudGuard"
                  size="small"
                  style={{ marginTop: 16, backgroundColor: `${meta.color}18`, color: meta.color, fontWeight: 600 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Layout>
  );
}
