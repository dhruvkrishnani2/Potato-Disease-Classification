import React, { useState } from "react";
import { useHistory, useLocation } from "react-router-dom";
import { makeStyles } from "@material-ui/core/styles";
import {
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  IconButton,
  Tooltip,
  Hidden,
  Box,
} from "@material-ui/core";
import DashboardIcon from "@material-ui/icons/Dashboard";
import CameraAltIcon from "@material-ui/icons/CameraAlt";
import HistoryIcon from "@material-ui/icons/History";
import MenuBookIcon from "@material-ui/icons/MenuBook";
import PersonIcon from "@material-ui/icons/Person";
import InfoIcon from "@material-ui/icons/Info";
import MenuIcon from "@material-ui/icons/Menu";
import ExitToAppIcon from "@material-ui/icons/ExitToApp";
import EcoIcon from "@material-ui/icons/Eco";
import { useAuth } from "../context/AuthContext";
import { colors } from "../theme";

const DRAWER_WIDTH = 260;

const useStyles = makeStyles((theme) => ({
  root: { display: "flex", minHeight: "100vh", backgroundColor: colors.background },
  appBar: {
    background: colors.gradient,
    boxShadow: "0 2px 12px rgba(27,67,50,0.2)",
    zIndex: theme.zIndex.drawer + 1,
  },
  toolbar: { ...theme.mixins.toolbar },
  drawer: {
    width: DRAWER_WIDTH,
    flexShrink: 0,
  },
  drawerPaper: {
    width: DRAWER_WIDTH,
    borderRight: "none",
    backgroundColor: colors.primaryDark,
    color: "#fff",
  },
  drawerHeader: {
    padding: theme.spacing(3, 2),
    display: "flex",
    alignItems: "center",
    gap: theme.spacing(1.5),
  },
  logoIcon: {
    backgroundColor: colors.accent,
    width: 44,
    height: 44,
  },
  logoText: { fontWeight: 800, fontSize: "1.25rem", letterSpacing: "-0.5px" },
  logoSub: { fontSize: "0.7rem", opacity: 0.7, marginTop: 2 },
  navItem: {
    borderRadius: 12,
    margin: theme.spacing(0.5, 1.5),
    "&:hover": { backgroundColor: "rgba(255,255,255,0.08)" },
  },
  navItemActive: {
    backgroundColor: "rgba(82,183,136,0.25) !important",
    borderLeft: `3px solid ${colors.accent}`,
    "& .MuiListItemIcon-root": { color: colors.accent },
    "& .MuiListItemText-primary": { fontWeight: 700, color: "#fff" },
  },
  navIcon: { color: "rgba(255,255,255,0.65)", minWidth: 40 },
  navText: { "& .MuiListItemText-primary": { fontSize: "0.9rem", color: "rgba(255,255,255,0.85)" } },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
    [theme.breakpoints.down("sm")]: { padding: theme.spacing(2) },
  },
  userSection: {
    marginTop: "auto",
    padding: theme.spacing(2),
    borderTop: "1px solid rgba(255,255,255,0.1)",
  },
  userRow: { display: "flex", alignItems: "center", gap: theme.spacing(1.5) },
  userName: { fontSize: "0.85rem", fontWeight: 600, lineHeight: 1.2 },
  userEmail: { fontSize: "0.7rem", opacity: 0.6 },
  grow: { flexGrow: 1 },
  pageTitle: { fontWeight: 700, fontSize: "1.1rem" },
}));

const navItems = [
  { label: "Dashboard", path: "/", icon: <DashboardIcon /> },
  { label: "Classify Leaf", path: "/classify", icon: <CameraAltIcon /> },
  { label: "Scan History", path: "/history", icon: <HistoryIcon /> },
  { label: "Disease Guide", path: "/diseases", icon: <MenuBookIcon /> },
  { label: "Profile", path: "/profile", icon: <PersonIcon /> },
  { label: "About", path: "/about", icon: <InfoIcon /> },
];

export default function Layout({ children, title }) {
  const classes = useStyles();
  const history = useHistory();
  const location = useLocation();
  const { user, logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const drawer = (
    <Box display="flex" flexDirection="column" height="100%">
      <div className={classes.drawerHeader}>
        <Avatar className={classes.logoIcon}>
          <EcoIcon />
        </Avatar>
        <div>
          <Typography className={classes.logoText}>SpudGuard</Typography>
          <Typography className={classes.logoSub}>Disease Detection</Typography>
        </div>
      </div>
      <List style={{ flex: 1 }}>
        {navItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <ListItem
              button
              key={item.path}
              className={`${classes.navItem} ${active ? classes.navItemActive : ""}`}
              onClick={() => {
                history.push(item.path);
                setMobileOpen(false);
              }}
            >
              <ListItemIcon className={classes.navIcon}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} className={classes.navText} />
            </ListItem>
          );
        })}
      </List>
      <div className={classes.userSection}>
        <div className={classes.userRow}>
          <Avatar src={user?.picture} alt={user?.name} style={{ width: 36, height: 36 }} />
          <div style={{ flex: 1, overflow: "hidden" }}>
            <Typography className={classes.userName} noWrap>
              {user?.name}
            </Typography>
            <Typography className={classes.userEmail} noWrap>
              {user?.email}
            </Typography>
          </div>
          <Tooltip title="Sign out">
            <IconButton size="small" onClick={logout} style={{ color: "rgba(255,255,255,0.6)" }}>
              <ExitToAppIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </div>
      </div>
    </Box>
  );

  return (
    <div className={classes.root}>
      <AppBar position="fixed" className={classes.appBar}>
        <Toolbar>
          <Hidden mdUp>
            <IconButton edge="start" color="inherit" onClick={() => setMobileOpen(!mobileOpen)}>
              <MenuIcon />
            </IconButton>
          </Hidden>
          <Typography className={classes.pageTitle}>{title}</Typography>
          <div className={classes.grow} />
          <Hidden smDown>
            <Avatar src={user?.picture} alt={user?.name} style={{ width: 32, height: 32 }} />
          </Hidden>
        </Toolbar>
      </AppBar>

      <Hidden mdUp>
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          classes={{ paper: classes.drawerPaper }}
        >
          {drawer}
        </Drawer>
      </Hidden>
      <Hidden smDown>
        <Drawer variant="permanent" classes={{ paper: classes.drawerPaper }} className={classes.drawer}>
          <div className={classes.toolbar} />
          {drawer}
        </Drawer>
      </Hidden>

      <main className={classes.content}>
        <div className={classes.toolbar} />
        {children}
      </main>
    </div>
  );
}
