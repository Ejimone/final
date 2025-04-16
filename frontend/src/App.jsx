import { useState } from "react";
import "./App.css";
import EmailComponent from "./components/email";
import AudioComponent from "./components/Audio";

function App() {
  const [activeTab, setActiveTab] = useState("email"); // Default to email tab

  // For debugging
  console.log("App rendering, activeTab:", activeTab);

  return (
    <div className="app-container">
      <header>
        <h1>Dashboard</h1>
        <nav>
          <button
            className={activeTab === "email" ? "active" : ""}
            onClick={() => setActiveTab("email")}
          >
            Email Assistant
          </button>
          <button
            className={activeTab === "voice" ? "active" : ""}
            onClick={() => setActiveTab("voice")}
          >
            Voice Agent
          </button>
        </nav>
      </header>

      <main>
        {/* Debugging message */}
        <div style={{ marginBottom: "10px", color: "blue" }}>
          Active Tab: {activeTab}
        </div>

        {activeTab === "email" ? <EmailComponent /> : <AudioComponent />}
      </main>

      <footer>
        <p>AI Assistant Dashboard © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
