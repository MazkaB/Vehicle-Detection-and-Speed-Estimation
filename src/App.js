import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import VideoDashboard from './VideoDashboard';
import VideoStream from './VideoStream';
import Statistics from './Statistics';

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<VideoDashboard />} />
        <Route path="/videos/:videoId" element={<VideoStream />} />
        <Route path="/videos/:videoId/stats" element={<Statistics />} />
      </Routes>
    </Router>
  );
}
