import { useState } from "react";
import axios from "axios";

export default function App() {
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRecord = async () => {
    setLoading(true);
    setEmotion(null);

    try {
      // Record audio from microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", audioBlob, "recorded_audio.wav");

        try {
          const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          setEmotion(response.data.emotion);
        } catch (error) {
          console.error("Error fetching emotion:", error);
          setEmotion("Error detecting emotion");
        }

        setLoading(false);
      };

      // Start recording and stop after 3 seconds
      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), 3000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      setEmotion("Microphone access denied");
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "100px" }}>
      <h1>Real-Time Emotion Detector</h1>
      <button
        onClick={handleRecord}
        style={{
          padding: "10px 20px",
          fontSize: "18px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          margin: "20px",
        }}
        disabled={loading}
      >
        {loading ? "Recording..." : "Record & Detect Emotion"}
      </button>
      {emotion && <h2 style={{ color: "blue" }}>Detected Emotion: {emotion}</h2>}
    </div>
  );
}
