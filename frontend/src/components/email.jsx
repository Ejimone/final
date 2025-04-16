import { useState } from "react";
import { sendEmail, generateEmailContent } from "../services/api";

const EmailComponent = () => {
  const [to, setTo] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerateEmail = async (e) => {
    e.preventDefault();
    if (!prompt) {
      setError("Please enter a prompt for email generation");
      return;
    }

    try {
      setGenerating(true);
      setError(null);

      const response = await generateEmailContent(prompt);

      if (response.status === "success") {
        setBody(response.content);
        setResult({ message: "Email content generated successfully" });
      } else {
        setError(response.message || "Failed to generate email content");
      }
    } catch (err) {
      setError(
        err.message || "An error occurred while generating email content"
      );
    } finally {
      setGenerating(false);
    }
  };

  const handleSendEmail = async (e) => {
    e.preventDefault();

    // Simple validation
    if (!to || !subject || !body) {
      setError("Please fill in all fields");
      return;
    }

    // Basic email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(to)) {
      setError("Please enter a valid email address");
      return;
    }

    try {
      setSending(true);
      setError(null);

      const emailData = { to, subject, body };
      const response = await sendEmail(emailData);

      if (response.status === "success") {
        setResult({
          message: "Email sent successfully",
          details: response.details,
        });

        // Clear form after successful send
        setTo("");
        setSubject("");
        setBody("");
        setPrompt("");
      } else {
        setError(response.message || "Failed to send email");
      }
    } catch (err) {
      setError(err.message || "An error occurred while sending the email");
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="email-container">
      <h2>Email Assistant</h2>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {result && (
        <div className="success-message">
          <p>{result.message}</p>
          {result.details && (
            <div>
              <p>Recipient: {result.details.to}</p>
              <p>Subject: {result.details.subject}</p>
            </div>
          )}
          <button onClick={() => setResult(null)}>Dismiss</button>
        </div>
      )}

      <div className="generate-section">
        <h3>Generate Email Content</h3>
        <form onSubmit={handleGenerateEmail}>
          <div className="form-group">
            <label htmlFor="prompt">
              Describe what you want in your email:
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows="3"
              placeholder="E.g., Write a professional email to schedule a meeting with my team about the upcoming project deadline"
            />
          </div>
          <button type="submit" disabled={generating || !prompt}>
            {generating ? "Generating..." : "Generate Email Content"}
          </button>
        </form>
      </div>

      <div className="send-section">
        <h3>Send Email</h3>
        <form onSubmit={handleSendEmail}>
          <div className="form-group">
            <label htmlFor="to">To:</label>
            <input
              type="email"
              id="to"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              placeholder="recipient@example.com"
            />
          </div>

          <div className="form-group">
            <label htmlFor="subject">Subject:</label>
            <input
              type="text"
              id="subject"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="Email subject"
            />
          </div>

          <div className="form-group">
            <label htmlFor="body">Body:</label>
            <textarea
              id="body"
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows="6"
              placeholder="Email body"
            />
          </div>

          <button type="submit" disabled={sending || !to || !subject || !body}>
            {sending ? "Sending..." : "Send Email"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default EmailComponent;
