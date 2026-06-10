import { BrowserRouter, Switch, Route, Redirect } from "react-router-dom";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { AuthProvider } from "./context/AuthContext";
import { HistoryProvider } from "./context/HistoryContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Dashboard from "./components/Dashboard";
import Classify from "./components/Classify";
import HistoryPage from "./components/History";
import DiseaseGuide from "./components/DiseaseGuide";
import Profile from "./components/Profile";
import About from "./components/About";

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || "";

function AppRoutes() {
  return (
    <Switch>
      <Route exact path="/" component={Dashboard} />
      <Route path="/classify" component={Classify} />
      <Route path="/history" component={HistoryPage} />
      <Route path="/diseases" component={DiseaseGuide} />
      <Route path="/profile" component={Profile} />
      <Route path="/about" component={About} />
      <Redirect to="/" />
    </Switch>
  );
}

function AppCore() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <ProtectedRoute>
          <HistoryProvider>
            <AppRoutes />
          </HistoryProvider>
        </ProtectedRoute>
      </BrowserRouter>
    </AuthProvider>
  );
}

function App() {
  if (GOOGLE_CLIENT_ID) {
    return (
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        <AppCore />
      </GoogleOAuthProvider>
    );
  }
  return <AppCore />;
}

export default App;
