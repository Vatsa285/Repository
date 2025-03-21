import React, { useState, useEffect } from "react";
import { ReactMic } from "react-mic";
import axios from "axios";
import "./index.css";

// Icons for emotions
const emotionIcons = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  neutral: "😐",
  surprised: "😲",
  fearful: "😨",
  disgusted: "🤢",
  calm:"😌"
};

// Descriptions for emotions
const emotionDescriptions = {
  happy: "You sound cheerful and positive!",
  sad: "Your voice carries a melancholic tone.",
  angry: "There's intensity and frustration in your voice.",
  neutral: "Your voice sounds balanced and composed.",
  surprised: "You sound astonished or taken aback!",
  fearful: "There's a sense of worry in your voice.",
  disgusted: "Your voice expresses aversion or dislike.",
  calm:"You sound peaceful!."
};

function App() {
  const [recording, setRecording] = useState(false);
  const [blob, setBlob] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [loading, setLoading] = useState(false);
  const [audioURL, setAudioURL] = useState("");
  const [recordingTime, setRecordingTime] = useState(0);
  const [hasRecorded, setHasRecorded] = useState(false);

  // Timer for recording duration
  useEffect(() => {
    let interval;
    if (recording) {
      interval = setInterval(() => {
        setRecordingTime((prevTime) => prevTime + 1);
      }, 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [recording]);

  const startRecording = () => {
    setRecording(true);
    setRecordingTime(0);
    setEmotion("");
  };

  const stopRecording = () => {
    setRecording(false);
    setHasRecorded(true);
  };

  const onStop = (recordedBlob) => {
    setBlob(recordedBlob.blob);
    setAudioURL(recordedBlob.blobURL);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  const sendAudio = async () => {
    if (!blob) {
        alert("Please record your voice first!");
        return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", blob, "audio.wav");

    try {
        const response = await axios.post("http://127.0.0.1:8000/predict/", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        });

        console.log("API Response:", response.data); // Debugging log

        if (response.data.status === "success") {
            setEmotion(response.data.emotion);  // Ensure this updates the UI
        } else {
            setEmotion("Error detecting emotion");
        }
    } catch (error) {
        console.error("Error sending audio:", error);
        alert("Error detecting emotion. Please try again.");
        setEmotion("Error");
    }

    setLoading(false);
};


  const playAudio = () => {
    if (audioURL) {
      new Audio(audioURL).play();
    }
  };

  return (
    <div className="container">
      <div className="app-header">
        <h1>
          <span>🎭</span> Voice Emotion Detector
        </h1>
        <p>
          Record your voice and our AI will detect the emotion in your speech
        </p>
      </div>

      <div className="content">
        <div className="visualizer-container">
          <ReactMic
            record={recording}
            className={`react-mic ${recording ? "react-mic-recording" : ""}`}
            onStop={onStop}
            mimeType="audio/wav"
            strokeColor="#6366f1"
            backgroundColor="#f3f4f6"
          />
          {recording && (
            <div className="recording-indicator">
              <div className="recording-dot"></div>
              Recording {formatTime(recordingTime)}
            </div>
          )}
        </div>

        <div className="controls">
          <button onClick={startRecording} className="btn btn-start" disabled={recording}>
            Start Recording
          </button>
          <button onClick={stopRecording} className="btn btn-stop" disabled={!recording}>
            Stop Recording
          </button>
          {hasRecorded && <button onClick={playAudio} className="btn btn-play">Play Recording</button>}
          <button onClick={sendAudio} className="btn btn-detect" disabled={loading || !blob}>
            {loading ? "Analyzing..." : "Detect Emotion"}
          </button>
        </div>

        {emotion && <div className="emotion-result">{emotionIcons[emotion]} {emotionDescriptions[emotion]}</div>}
      </div>
    </div>
  );
}

export default App;
