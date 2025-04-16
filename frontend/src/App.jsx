import { useState } from "react";
import "./App.css";
import EmailComponent from "./components/email";
import AudioComponent from "./components/Audio";

function App() {
  const [activeTab, setActiveTab] = useState("email"); // Default to email tab

  return (
    <div className="app-container">
      <header>
        <h1>AI Assistant Dashboard</h1>
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
        {activeTab === "email" ? <EmailComponent /> : <AudioComponent />}
      </main>

      <footer>
        <p>AI Assistant Dashboard © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
