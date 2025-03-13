import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import PresenceDashboard from './pages/presenceDashboard';
import FaceRegistration from './pages/faceRegistration'; // You'll need to create this component

function App() {
  return (
    <Router>
      <div className="App">
        <header className="app-header">
          <h1 className="app-title">Face Recognition System</h1>
          <nav className="nav-buttons">
            <Link to="/" className="nav-button">Presence Dashboard</Link>
            <Link to="/registration" className="nav-button">Face Registration</Link>
          </nav>
        </header>
        <main className="app-content">
          <Routes>
            <Route path="/" element={<PresenceDashboard />} />
            <Route path="/registration" element={<FaceRegistration />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;