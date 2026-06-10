import React, { useState, useEffect } from "react";
import { makeStyles } from "@material-ui/core/styles";
import {
  Grid,
  Card,
  CardContent,
  CardMedia,
  CardActionArea,
  Typography,
  Button,
  CircularProgress,
  Chip,
  Box,
  Paper,
} from "@material-ui/core";
import CloudUploadIcon from "@material-ui/icons/CloudUpload";
import ClearIcon from "@material-ui/icons/Clear";
import SmartToyIcon from "@material-ui/icons/Android";
import { DropzoneArea } from "material-ui-dropzone";
import axios from "axios";
import Layout from "./Layout";
import { useAuth } from "../context/AuthContext";
import { useHistory } from "../context/HistoryContext";
import { colors, diseaseMeta, getDiseaseColor } from "../theme";

const useStyles = makeStyles((theme) => ({
  uploadCard: {
    borderRadius: 20,
    boxShadow: colors.cardShadow,
    overflow: "hidden",
    minHeight: 320,
  },
  uploadZone: {
    padding: theme.spacing(2),
    "& .MuiDropzoneArea-root": {
      backgroundColor: `${colors.primary}08 !important`,
      border: `2px dashed ${colors.accent} !important`,
      borderRadius: "16px !important",
      minHeight: "260px !important",
    },
    "& .MuiDropzoneArea-text": {
      color: `${colors.text} !important`,
      fontSize: "0.95rem !important",
      marginTop: "12px !important",
    },
    "& .MuiDropzoneArea-icon": {
      color: `${colors.primaryLight} !important`,
    },
  },
  previewMedia: { height: 280, objectFit: "cover" },
  resultCard: {
    borderRadius: 20,
    boxShadow: colors.cardShadow,
    textAlign: "center",
    padding: theme.spacing(3),
  },
  resultIcon: { fontSize: "3.5rem", marginBottom: theme.spacing(1) },
  resultLabel: { fontWeight: 800, fontSize: "1.5rem", marginBottom: theme.spacing(0.5) },
  confidenceRing: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: 80,
    height: 80,
    borderRadius: "50%",
    border: `4px solid ${colors.accent}`,
    margin: theme.spacing(2, "auto"),
    fontWeight: 800,
    fontSize: "1.1rem",
    color: colors.primary,
  },
  adviceCard: {
    borderRadius: 20,
    boxShadow: colors.cardShadow,
    borderLeft: `4px solid ${colors.accent}`,
  },
  adviceHeader: {
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(1),
    marginBottom: theme.spacing(2),
    color: colors.primary,
    fontWeight: 700,
  },
  adviceContent: {
    whiteSpace: "pre-wrap",
    lineHeight: 1.8,
    color: colors.text,
    fontSize: "0.92rem",
  },
  clearBtn: {
    borderRadius: 12,
    textTransform: "none",
    fontWeight: 700,
    padding: "10px 32px",
    borderColor: colors.primaryLight,
    color: colors.primaryLight,
  },
  loadingBox: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: theme.spacing(6),
    gap: theme.spacing(2),
  },
  stepIndicator: {
    display: "flex",
    gap: theme.spacing(1),
    marginBottom: theme.spacing(3),
    justifyContent: "center",
  },
  step: {
    padding: "6px 16px",
    borderRadius: 20,
    fontSize: "0.8rem",
    fontWeight: 600,
    backgroundColor: colors.background,
    color: colors.textMuted,
  },
  stepActive: {
    backgroundColor: colors.primary,
    color: "#fff",
  },
}));

export default function Classify() {
  const classes = useStyles();
  const { authHeaders, API_BASE } = useAuth();
  const { addScan } = useHistory();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [aiAdvice, setAiAdvice] = useState();
  const [aiLoading, setAiLoading] = useState(false);
  const [image, setImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const sendFile = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const res = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { ...authHeaders(), "Content-Type": "multipart/form-data" },
      });
      if (res.status === 200) {
        setData(res.data);
        fetchAiAdvice(res.data.class, res.data.confidence);
      }
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAiAdvice = async (disease, conf) => {
    setAiLoading(true);
    setAiAdvice(null);
    try {
      const res = await axios.post(
        `${API_BASE}/ai/advice`,
        { disease, confidence: conf },
        { headers: authHeaders() }
      );
      setAiAdvice(res.data);
      addScan({ disease, confidence: conf, adviceSource: res.data.source });
    } catch (err) {
      console.error("AI advice failed:", err);
      addScan({ disease, confidence: conf });
    } finally {
      setAiLoading(false);
    }
  };

  const clearData = () => {
    setData(null);
    setAiAdvice(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) return;
    setIsLoading(true);
    sendFile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      setAiAdvice(null);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setAiAdvice(null);
    setImage(true);
  };

  const confidence = data ? (parseFloat(data.confidence) * 100).toFixed(1) : 0;
  const meta = data ? diseaseMeta[data.class] : null;
  const currentStep = !image ? 1 : isLoading ? 2 : data ? 3 : 1;

  return (
    <Layout title="Classify Leaf">
      <div className={classes.stepIndicator}>
        {["Upload", "Analyze", "Results"].map((label, i) => (
          <span key={label} className={`${classes.step} ${currentStep === i + 1 ? classes.stepActive : ""}`}>
            {i + 1}. {label}
          </span>
        ))}
      </div>

      <Grid container spacing={3} justifyContent="center">
        <Grid item xs={12} md={5}>
          <Card className={classes.uploadCard}>
            {image && preview ? (
              <CardActionArea>
                <CardMedia className={classes.previewMedia} image={preview} component="img" title="Uploaded leaf" />
              </CardActionArea>
            ) : (
              <CardContent className={classes.uploadZone}>
                <Box textAlign="center" mb={2}>
                  <CloudUploadIcon style={{ fontSize: 48, color: colors.primaryLight }} />
                  <Typography variant="h6" style={{ fontWeight: 700, color: colors.text, marginTop: 8 }}>
                    Upload Leaf Image
                  </Typography>
                  <Typography variant="body2" style={{ color: colors.textMuted }}>
                    JPG, PNG — clear photo of a potato plant leaf
                  </Typography>
                </Box>
                <DropzoneArea
                  acceptedFiles={["image/*"]}
                  dropzoneText="Drag & drop or click to browse"
                  onChange={onSelectFile}
                  showPreviews={false}
                  showAlerts={false}
                />
              </CardContent>
            )}
            {isLoading && (
              <div className={classes.loadingBox}>
                <CircularProgress style={{ color: colors.primaryLight }} />
                <Typography style={{ fontWeight: 600, color: colors.text }}>Analyzing leaf...</Typography>
              </div>
            )}
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          {data && (
            <Paper className={classes.resultCard} elevation={0}>
              <div className={classes.resultIcon}>{meta?.icon}</div>
              <Typography
                className={classes.resultLabel}
                style={{ color: getDiseaseColor(data.class) }}
              >
                {data.class}
              </Typography>
              <div className={classes.confidenceRing}>{confidence}%</div>
              <Typography variant="body2" style={{ color: colors.textMuted }}>
                Model confidence
              </Typography>
              <Button
                variant="outlined"
                className={classes.clearBtn}
                startIcon={<ClearIcon />}
                onClick={clearData}
                style={{ marginTop: 24 }}
              >
                Scan Another
              </Button>
            </Paper>
          )}
          {!data && !isLoading && image && (
            <Paper className={classes.resultCard} elevation={0}>
              <CircularProgress style={{ color: colors.primaryLight }} />
            </Paper>
          )}
          {!image && (
            <Paper
              className={classes.resultCard}
              elevation={0}
              style={{ backgroundColor: colors.background, minHeight: 200, display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <Typography style={{ color: colors.textMuted }}>Results will appear here</Typography>
            </Paper>
          )}
        </Grid>

        {aiLoading && (
          <Grid item xs={12}>
            <Box textAlign="center" py={2}>
              <CircularProgress size={24} style={{ color: colors.primaryLight }} />
              <Typography variant="body2" style={{ marginTop: 8, color: colors.textMuted }}>
                Generating AI treatment advice...
              </Typography>
            </Box>
          </Grid>
        )}

        {aiAdvice && (
          <Grid item xs={12} md={10}>
            <Card className={classes.adviceCard}>
              <CardContent>
                <Typography variant="h6" className={classes.adviceHeader}>
                  <SmartToyIcon /> AI Treatment Advice
                  <Chip
                    size="small"
                    label={aiAdvice.source === "gemini" ? "Powered by Gemini" : "Built-in guide"}
                    style={{ backgroundColor: "#d8f3dc", color: colors.primary, fontWeight: 600 }}
                  />
                </Typography>
                <Typography className={classes.adviceContent}>{aiAdvice.advice}</Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Layout>
  );
}
