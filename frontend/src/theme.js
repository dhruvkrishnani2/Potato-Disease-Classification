export const colors = {
  primary: "#1b4332",
  primaryLight: "#2d6a4f",
  primaryDark: "#081c15",
  accent: "#52b788",
  accentLight: "#95d5b2",
  surface: "#ffffff",
  background: "#f0f7f4",
  text: "#1a1a2e",
  textMuted: "#6b7280",
  earlyBlight: "#e76f51",
  lateBlight: "#c1121f",
  healthy: "#2d6a4f",
  warning: "#f4a261",
  gradient: "linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%)",
  cardShadow: "0 4px 24px rgba(27, 67, 50, 0.08)",
  cardShadowHover: "0 8px 32px rgba(27, 67, 50, 0.14)",
};

export const diseaseMeta = {
  "Early Blight": {
    color: colors.earlyBlight,
    icon: "🍂",
    description: "Brown spots with concentric rings on older leaves",
  },
  "Late Blight": {
    color: colors.lateBlight,
    icon: "🦠",
    description: "Water-soaked lesions spreading rapidly in humid weather",
  },
  Healthy: {
    color: colors.healthy,
    icon: "🌿",
    description: "No significant disease symptoms detected",
  },
};

export function getDiseaseColor(disease) {
  return diseaseMeta[disease]?.color || colors.textMuted;
}
