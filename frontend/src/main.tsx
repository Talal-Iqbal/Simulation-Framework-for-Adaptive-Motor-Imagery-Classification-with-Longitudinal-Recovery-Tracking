import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { SessionStoreProvider } from "./state/sessionStore";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <SessionStoreProvider>
      <App />
    </SessionStoreProvider>
  </React.StrictMode>
);
