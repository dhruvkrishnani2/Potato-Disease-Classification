import React from "react";
import { useHistory as useRouter } from "react-router-dom";
import { makeStyles } from "@material-ui/core/styles";
import {
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Box,
} from "@material-ui/core";
import DeleteSweepIcon from "@material-ui/icons/DeleteSweep";
import CameraAltIcon from "@material-ui/icons/CameraAlt";
import Layout from "./Layout";
import { useHistory } from "../context/HistoryContext";
import { colors, diseaseMeta, getDiseaseColor } from "../theme";

const useStyles = makeStyles((theme) => ({
  card: {
    borderRadius: 20,
    boxShadow: colors.cardShadow,
  },
  tableHead: {
    backgroundColor: colors.background,
    "& th": { fontWeight: 700, color: colors.text, fontSize: "0.82rem", borderBottom: "none" },
  },
  tableRow: {
    "&:hover": { backgroundColor: `${colors.primary}06` },
    "& td": { borderColor: colors.background },
  },
  emptyState: {
    textAlign: "center",
    padding: theme.spacing(6),
    color: colors.textMuted,
  },
  clearBtn: {
    borderRadius: 10,
    textTransform: "none",
    fontWeight: 600,
    color: colors.lateBlight,
    borderColor: colors.lateBlight,
  },
  ctaBtn: {
    borderRadius: 10,
    textTransform: "none",
    fontWeight: 700,
    background: colors.gradient,
    color: "#fff",
    marginTop: theme.spacing(2),
    "&:hover": { background: colors.primary },
  },
}));

export default function HistoryPage() {
  const classes = useStyles();
  const router = useRouter();
  const { history: scans, clearHistory } = useHistory();

  return (
    <Layout title="Scan History">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="body2" style={{ color: colors.textMuted }}>
          {scans.length} scan{scans.length !== 1 ? "s" : ""} recorded
        </Typography>
        {scans.length > 0 && (
          <Button
            variant="outlined"
            size="small"
            className={classes.clearBtn}
            startIcon={<DeleteSweepIcon />}
            onClick={clearHistory}
          >
            Clear History
          </Button>
        )}
      </Box>

      <Card className={classes.card}>
        <CardContent style={{ padding: 0 }}>
          {scans.length === 0 ? (
            <div className={classes.emptyState}>
              <Typography variant="h3" style={{ marginBottom: 8 }}>
                📋
              </Typography>
              <Typography variant="h6" style={{ fontWeight: 700, color: colors.text }}>
                No scan history yet
              </Typography>
              <Typography variant="body2">Your classification results will be saved here automatically.</Typography>
              <Button
                className={classes.ctaBtn}
                startIcon={<CameraAltIcon />}
                onClick={() => router.push("/classify")}
              >
                Classify a Leaf
              </Button>
            </div>
          ) : (
            <TableContainer>
              <Table>
                <TableHead className={classes.tableHead}>
                  <TableRow>
                    <TableCell>Date & Time</TableCell>
                    <TableCell>Diagnosis</TableCell>
                    <TableCell align="center">Confidence</TableCell>
                    <TableCell align="center">AI Source</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {scans.map((scan) => (
                    <TableRow key={scan.id} className={classes.tableRow}>
                      <TableCell>
                        <Typography variant="body2" style={{ fontWeight: 500 }}>
                          {new Date(scan.timestamp).toLocaleDateString()}
                        </Typography>
                        <Typography variant="caption" style={{ color: colors.textMuted }}>
                          {new Date(scan.timestamp).toLocaleTimeString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center" style={{ gap: 8 }}>
                          <span>{diseaseMeta[scan.disease]?.icon}</span>
                          <Typography style={{ fontWeight: 600, color: getDiseaseColor(scan.disease) }}>
                            {scan.disease}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={`${(scan.confidence * 100).toFixed(1)}%`}
                          size="small"
                          style={{
                            fontWeight: 700,
                            backgroundColor: `${getDiseaseColor(scan.disease)}15`,
                            color: getDiseaseColor(scan.disease),
                          }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={scan.adviceSource === "gemini" ? "Gemini AI" : scan.adviceSource || "—"}
                          size="small"
                          variant="outlined"
                          style={{ fontSize: "0.75rem" }}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Layout>
  );
}
