import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import * as faceapi from 'face-api.js';

function FaceRegistration() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedAngles, setCapturedAngles] = useState([]);
  const [formData, setFormData] = useState({ name: '', id: '' });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [detectionInProgress, setDetectionInProgress] = useState(false);
  const [anglesToCapture, setAnglesToCapture] = useState([
    { name: 'frontal', captured: false, label: 'Look directly at the camera', threshold: 10 },
    { name: 'slight_left', captured: false, label: 'Turn slightly to the left', threshold: 25 },
    { name: 'profile_left', captured: false, label: 'Turn more to the left', threshold: 45 },
    { name: 'slight_right', captured: false, label: 'Turn slightly to the right', threshold: -25 },
    { name: 'profile_right', captured: false, label: 'Turn more to the right', threshold: -45 },
    { name: 'tilted_up', captured: false, label: 'Tilt your head up slightly', threshold: 15 },
    { name: 'tilted_down', captured: false, label: 'Tilt your head down slightly', threshold: -15 }
  ]);
  const [currentGuideText, setCurrentGuideText] = useState('');
  const [captureDelay, setCaptureDelay] = useState(false);
  const [isFaceDetected, setIsFaceDetected] = useState(false); // New state for face detection

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const displayCanvasRef = useRef(null);

  // Initialize webcam and load models
  useEffect(() => {
    let stream = null;

    const setupCamera = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        ]);

        setMessage('Models loaded successfully');

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240 },     
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsCameraOn(true);
        }
      } catch (err) {
        setError(`Camera or model error: ${err.message}`);
        console.error("Error setting up camera or models:", err);
      }
    };

    setupCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Handle input changes
  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // Capture frame and return base64 image
  const captureFrame = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) {
      setError('Camera not initialized properly');
      return null;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (video.paused || video.ended || !video.srcObject) {
      setError('Video stream not available');
      return null;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      setError('Could not get canvas context');
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.9);
  }, []);

  // Calculate face orientation
  const calculateFaceOrientation = useCallback((landmarks) => {
    if (!landmarks) return null;

    const nose = landmarks.getNose();
    const jawOutline = landmarks.getJawOutline();
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    const faceCenter = {
      x: (jawOutline[0].x + jawOutline[16].x) / 2,
      y: (jawOutline[0].y + jawOutline[16].y) / 2
    };

    const eyeCenter = {
      x: (leftEye[0].x + rightEye[3].x) / 2,
      y: (leftEye[0].y + rightEye[3].y) / 2
    };

    const nosePos = nose[3];

    // Fix: Define eyeLine based on eye positions
    const eyeLine = {
      x: eyeCenter.x,
      y: eyeCenter.y
    };

    const horizontalAngle = ((nosePos.x - eyeCenter.x) / (rightEye[3].x - leftEye[0].x)) * 100;
    const verticalAngle = ((eyeLine.y - nose[3].y) / (landmarks.getJawOutline()[8].y - eyeLine.y)) * 30;

    return {
      horizontal: horizontalAngle,
      vertical: verticalAngle
    };
  }, []);

  // Continuously run face detection when capturing
  useEffect(() => {
    let animationFrameId;
    let lastCaptureTime = 0;
    const CAPTURE_COOLDOWN = 3000;

    const detectFaces = async () => {
      if (!isCapturing || !videoRef.current || !displayCanvasRef.current) {
        return;
      }

      try {
        setDetectionInProgress(true);

        const video = videoRef.current;
        const displayCanvas = displayCanvasRef.current;
        const displayContext = displayCanvas.getContext('2d');

        displayCanvas.width = video.videoWidth;
        displayCanvas.height = video.videoHeight;

        displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);

        // Draw circular mask
        displayContext.beginPath();
        displayContext.arc(displayCanvas.width / 2, displayCanvas.height / 2, 100, 0, 2 * Math.PI);
        displayContext.strokeStyle = isFaceDetected ? '#00ff00' : '#ff0000'; // Change border color based on face detection
        displayContext.lineWidth = 2;
        displayContext.stroke();

        // Draw the current video frame
        displayContext.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);

        const detections = await faceapi.detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions()
        ).withFaceLandmarks();

        if (detections.length === 0) {
          setIsFaceDetected(false); // No face detected
          setCurrentGuideText("No face detected. Please position yourself in the frame.");
          animationFrameId = requestAnimationFrame(detectFaces);
          return;
        }

        setIsFaceDetected(true); // Face detected
        faceapi.draw.drawFaceLandmarks(displayCanvas, detections);

        const landmarks = detections[0].landmarks;
        const orientation = calculateFaceOrientation(landmarks);

        if (!orientation) {
          setCurrentGuideText("Cannot determine face orientation. Please adjust position.");
          animationFrameId = requestAnimationFrame(detectFaces);
          return;
        }

        const now = Date.now();
        const canCapture = now - lastCaptureTime > CAPTURE_COOLDOWN && !captureDelay;

        const remainingAngles = anglesToCapture.filter(angle => !angle.captured);

        if (remainingAngles.length === 0) {
          setCurrentGuideText("All angles captured successfully!");
          setIsCapturing(false);
          return;
        }

        const nextAngleIndex = anglesToCapture.findIndex(angle => !angle.captured);
        const nextAngle = anglesToCapture[nextAngleIndex];

        if (nextAngle.name.includes('left') || nextAngle.name.includes('right')) {
          setCurrentGuideText(`${nextAngle.label} (Current horizontal angle: ${orientation.horizontal.toFixed(1)}°)`);
        } else if (nextAngle.name.includes('up') || nextAngle.name.includes('down')) {
          setCurrentGuideText(`${nextAngle.label} (Current vertical angle: ${orientation.vertical.toFixed(1)}°)`);
        } else {
          setCurrentGuideText(`${nextAngle.label}`);
        }

        let matched = false;

        for (let i = 0; i < anglesToCapture.length; i++) {
          const angle = anglesToCapture[i];
          if (angle.captured) continue;

          let isCorrectPosition = false;

          if (angle.name === 'frontal') {
            isCorrectPosition = Math.abs(orientation.horizontal) < angle.threshold &&
              Math.abs(orientation.vertical) < 10;
          } else if (angle.name.includes('left')) {
            isCorrectPosition = orientation.horizontal > angle.threshold &&
              Math.abs(orientation.vertical) < 15;
          } else if (angle.name.includes('right')) {
            isCorrectPosition = orientation.horizontal < angle.threshold &&
              Math.abs(orientation.vertical) < 15;
          } else if (angle.name === 'tilted_up') {
            isCorrectPosition = orientation.vertical > angle.threshold &&
              Math.abs(orientation.horizontal) < 15;
          } else if (angle.name === 'tilted_down') {
            isCorrectPosition = orientation.vertical < angle.threshold &&
              Math.abs(orientation.horizontal) < 15;
          }

          if (isCorrectPosition && canCapture) {
            matched = true;

            const imageData = captureFrame();
            if (!imageData) {
              setError('Failed to capture image');
              continue;
            }

            try {
              setCaptureDelay(true);
              const response = await axios.post('http://localhost:5000/api/register-face', {
                image: imageData,
                personName: formData.name,
                personId: formData.id,
              });

              if (response.data.status === 'success') {
                setAnglesToCapture(prev => {
                  const newAngles = [...prev];
                  newAngles[i].captured = true;
                  return newAngles;
                });

                setCapturedAngles(prev => [...prev, angle.name]);

                setMessage(`Captured angle: ${angle.name}. ${remainingAngles.length - 1} angles remaining.`);

                lastCaptureTime = Date.now();

                setTimeout(() => setCaptureDelay(false), 1500);
                break;
              } else {
                setError(response.data.message || 'Unknown error occurred');
              }
            } catch (err) {
              setError(`Failed to register face: ${err.response?.data?.error || err.message}`);
              setTimeout(() => setCaptureDelay(false), 1500);
            }
          }
        }

        const allCaptured = anglesToCapture.every(angle => angle.captured);
        if (allCaptured) {
          setIsCapturing(false);
          setCurrentGuideText("All angles captured successfully!");
          setMessage("Face registration complete. Thank you!");
          return;
        }
      } catch (err) {
        console.error("Error in face detection:", err);
        setError(`Error in face detection: ${err.message}`);
      } finally {
        setDetectionInProgress(false);
        if (isCapturing) {
          animationFrameId = requestAnimationFrame(detectFaces);
        }
      }
    };

    if (isCapturing) {
      detectFaces();
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isCapturing, anglesToCapture, captureDelay, formData, calculateFaceOrientation, captureFrame, isFaceDetected]);

  // Start face registration
  const startRegistration = () => {
    if (!formData.name || !formData.id) {
      setError('Please fill out all fields');
      return;
    }

    setCapturedAngles([]);
    setAnglesToCapture(prev => prev.map(angle => ({ ...angle, captured: false })));
    setError('');
    setMessage('Starting face registration. Please follow the on-screen instructions.');
    setCurrentGuideText('Initializing face detection...');
    setIsCapturing(true);
  };

  // Stop face registration
  const stopRegistration = () => {
    setIsCapturing(false);
    setCurrentGuideText('');
    setMessage('Face registration stopped.');
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Intelligent Face Registration</h1>
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl">
        {error && <div className="text-red-500 mb-4">{error}</div>}
        {message && <div className="text-blue-600 mb-4">{message}</div>}

        {currentGuideText && (
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
            <p className="text-lg font-semibold text-blue-800">{currentGuideText}</p>
          </div>
        )}

        <div className="relative w-64 h-64 bg-gray-200 rounded-full overflow-hidden mb-6 mx-auto">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          <canvas
            ref={displayCanvasRef}
            className="absolute top-0 left-0 w-full h-full"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{ display: 'none' }}
          />
          {!isCameraOn && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <p className="text-white text-lg">Camera is initializing...</p>
            </div>
          )}
        </div>

        <div className="mb-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Capture Progress:</h3>
          <div className="grid grid-cols-7 gap-2">
            {anglesToCapture.map((angle, index) => (
              <div
                key={index}
                className={`p-2 rounded text-center text-sm ${
                  angle.captured
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {angle.name.replace('_', ' ')}
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="form-group">
            <label htmlFor="name" className="block text-gray-700">
              Full Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              placeholder="Enter your full name"
              disabled={isCapturing}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="form-group">
            <label htmlFor="id" className="block text-gray-700">
              ID Number
            </label>
            <input
              type="text"
              id="id"
              name="id"
              value={formData.id}
              onChange={handleInputChange}
              placeholder="Enter your ID number"
              disabled={isCapturing}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {!isCapturing ? (
            <button
              className="w-full bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed"
              onClick={startRegistration}
              disabled={!formData.name || !formData.id || !isCameraOn}
            >
              Start Face Registration
            </button>
          ) : (
            <button
              className="w-full bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 transition duration-200"
              onClick={stopRegistration}
            >
              Stop Registration
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default FaceRegistration;