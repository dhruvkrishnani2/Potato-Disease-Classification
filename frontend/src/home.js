import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import {
  Paper,
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
  IconButton,
  Tooltip,
  Chip,
} from "@material-ui/core";
import ExitToApp from "@material-ui/icons/ExitToApp";
import SmartToy from "@material-ui/icons/Android";
import image from "./bg.png";
import { DropzoneArea } from "material-ui-dropzone";
import { common } from "@material-ui/core/colors";
import Clear from "@material-ui/icons/Clear";
import axios from "axios";
import { useAuth } from "./context/AuthContext";

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    "&:hover": {
      backgroundColor: "#ffffff7a",
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  grow: { flexGrow: 1 },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  media: { height: 400 },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  mainContainer: {
    backgroundImage: `url(${image})`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundSize: "cover",
    minHeight: "93vh",
    marginTop: "8px",
    paddingBottom: theme.spacing(4),
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    height: 500,
    backgroundColor: "transparent",
    boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 30%) !important",
    borderRadius: "15px",
  },
  imageCardEmpty: { height: "auto" },
  detail: {
    backgroundColor: "white",
    display: "flex",
    justifyContent: "center",
    flexDirection: "column",
    alignItems: "center",
  },
  tableContainer: {
    backgroundColor: "transparent !important",
    boxShadow: "none !important",
  },
  table: { backgroundColor: "transparent !important" },
  tableHead: { backgroundColor: "transparent !important" },
  tableRow: { backgroundColor: "transparent !important" },
  tableCell: {
    fontSize: "22px",
    backgroundColor: "transparent !important",
    borderColor: "transparent !important",
    color: "#000000a6 !important",
    fontWeight: "bolder",
    padding: "1px 24px 1px 16px",
  },
  tableCell1: {
    fontSize: "14px",
    backgroundColor: "transparent !important",
    borderColor: "transparent !important",
    color: "#000000a6 !important",
    fontWeight: "bolder",
    padding: "1px 24px 1px 16px",
  },
  tableBody: { backgroundColor: "transparent !important" },
  buttonGrid: { maxWidth: "416px", width: "100%" },
  appbar: {
    background: "#2d6a4f",
    boxShadow: "none",
    color: "white",
  },
  loader: { color: "#2d6a4f !important" },
  userName: {
    marginRight: theme.spacing(1),
    display: { xs: "none", sm: "block" },
    fontSize: "0.9rem",
  },
  adviceCard: {
    margin: "auto",
    maxWidth: 600,
    marginTop: theme.spacing(2),
    borderRadius: "15px",
    boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 25%) !important",
  },
  adviceContent: {
    whiteSpace: "pre-wrap",
    lineHeight: 1.7,
    color: "#333",
    fontSize: "0.95rem",
  },
  adviceHeader: {
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(1),
    marginBottom: theme.spacing(2),
    color: "#2d6a4f",
    fontWeight: 700,
  },
  aiChip: { marginLeft: theme.spacing(1) },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const { user, logout, authHeaders, API_BASE } = useAuth();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [aiAdvice, setAiAdvice] = useState();
  const [aiLoading, setAiLoading] = useState(false);
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  let confidence = 0;

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
      setIsloading(false);
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
    } catch (err) {
      console.error("AI advice failed:", err);
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
    setIsloading(true);
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

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography variant="h6" noWrap>
            SpudGuard: Potato Disease Classification
          </Typography>
          <div className={classes.grow} />
          <Typography className={classes.userName}>{user?.name}</Typography>
          <Avatar src={user?.picture} alt={user?.name} style={{ marginRight: 8 }} />
          <Tooltip title="Sign out">
            <IconButton color="inherit" onClick={logout} size="small">
              <ExitToApp />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>
      <Container maxWidth={false} className={classes.mainContainer} disableGutters>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ""}`}>
              {image && (
                <CardActionArea>
                  <CardMedia className={classes.media} image={preview} component="img" title="Uploaded leaf" />
                </CardActionArea>
              )}
              {!image && (
                <CardContent>
                  <DropzoneArea
                    acceptedFiles={["image/*"]}
                    dropzoneText="Drag and drop a potato leaf image to classify"
                    onChange={onSelectFile}
                  />
                </CardContent>
              )}
              {data && (
                <CardContent className={classes.detail}>
                  <TableContainer component={Paper} className={classes.tableContainer}>
                    <Table className={classes.table} size="small">
                      <TableHead className={classes.tableHead}>
                        <TableRow className={classes.tableRow}>
                          <TableCell className={classes.tableCell1}>Label:</TableCell>
                          <TableCell align="right" className={classes.tableCell1}>
                            Confidence:
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody className={classes.tableBody}>
                        <TableRow className={classes.tableRow}>
                          <TableCell component="th" scope="row" className={classes.tableCell}>
                            {data.class}
                          </TableCell>
                          <TableCell align="right" className={classes.tableCell}>
                            {confidence}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              )}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress color="inherit" className={classes.loader} />
                  <Typography variant="h6" noWrap>
                    Classifying...
                  </Typography>
                </CardContent>
              )}
            </Card>
          </Grid>

          {aiLoading && (
            <Grid item xs={12} style={{ textAlign: "center" }}>
              <CircularProgress className={classes.loader} size={28} />
              <Typography variant="body2" style={{ marginTop: 8, color: "white" }}>
                Generating AI treatment advice...
              </Typography>
            </Grid>
          )}

          {aiAdvice && (
            <Grid item xs={12}>
              <Card className={classes.adviceCard}>
                <CardContent>
                  <Typography variant="h6" className={classes.adviceHeader}>
                    <SmartToy /> AI Treatment Advice
                    <Chip
                      size="small"
                      label={aiAdvice.source === "gemini" ? "Powered by Gemini" : "Built-in guide"}
                      className={classes.aiChip}
                      style={{ backgroundColor: "#d8f3dc", color: "#2d6a4f" }}
                    />
                  </Typography>
                  <Typography className={classes.adviceContent}>{aiAdvice.advice}</Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton
                variant="contained"
                className={classes.clearButton}
                color="primary"
                size="large"
                onClick={clearData}
                startIcon={<Clear fontSize="large" />}
              >
                Clear
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};
