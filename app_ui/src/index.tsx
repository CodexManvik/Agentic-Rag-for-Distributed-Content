import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Desktop } from "./screens/Desktop";

const rootEl = document.getElementById("app");

if (!rootEl) {
  throw new Error("Root element with id 'app' was not found");
}

createRoot(rootEl).render(
  <StrictMode>
    <Desktop />
  </StrictMode>,
);