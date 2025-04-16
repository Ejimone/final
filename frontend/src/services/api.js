// API service for communicating with the backend
const API_BASE_URL = "http://localhost:8000";

// Email API functions
export const sendEmail = async (emailData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/email/send`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(emailData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to send email");
    }

    return await response.json();
  } catch (error) {
    console.error("Error sending email:", error);
    throw error;
  }
};

export const generateEmailContent = async (prompt) => {
  try {
    const response = await fetch(`${API_BASE_URL}/email/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to generate email content");
    }

    return await response.json();
  } catch (error) {
    console.error("Error generating email content:", error);
    throw error;
  }
};

// Voice agent API functions
export const startVoiceAgent = async (roomName) => {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ room_name: roomName }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to start voice agent");
    }

    return await response.json();
  } catch (error) {
    console.error("Error starting voice agent:", error);
    throw error;
  }
};

export const makeOutboundCall = async (callData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/call/outbound`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(callData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to make outbound call");
    }

    return await response.json();
  } catch (error) {
    console.error("Error making outbound call:", error);
    throw error;
  }
};

// Time API functions
export const getCurrentTime = async (location) => {
  try {
    const response = await fetch(`${API_BASE_URL}/time/current`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ location }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to get current time");
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting current time:", error);
    throw error;
  }
};
