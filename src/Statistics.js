import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom'; // NEW OR MODIFIED
import { fetchVideoStats } from './api';
import { Bar } from 'react-chartjs-2';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function Statistics() {
  const { videoId } = useParams();
  const [stats, setStats] = useState(null);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 5000);
    return () => clearInterval(interval);
  }, [videoId]);

  async function loadStats() {
    const data = await fetchVideoStats(videoId);
    setStats(data);
  }

  if (!stats) return <div className="container">Loading...</div>;

  // Prepare chart data
  const classData = [
    { label: 'Car', value: stats.count_car },
    { label: 'Truck', value: stats.count_truck },
    { label: 'Motorcycle', value: stats.count_motorcycle },
    { label: 'Bus', value: stats.count_bus },
  ];

  const barData = {
    labels: classData.map((c) => c.label),
    datasets: [
      {
        label: 'Vehicles',
        data: classData.map((c) => c.value),
        backgroundColor: 'rgba(75,192,192,0.6)',
      },
    ],
  };

  const barOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Vehicle Class Count',
      },
    },
  };

  return (
    <div className="container">
      {/* Action buttons at the top */}
      <div className="d-flex justify-content-end mb-3"> {/* NEW OR MODIFIED */}
        <Link to={`/videos/${videoId}`} className="btn btn-success me-2">Back to Live</Link> {/* NEW OR MODIFIED */}
        <Link to="/" className="btn btn-secondary">Back to Dashboard</Link> {/* NEW OR MODIFIED */}
      </div>

      <h2>Statistics for Video {videoId}</h2>

      <div className="mb-3">
        <p><strong>Max Speed: </strong>{stats.max_speed.toFixed(2)} km/h</p>
        <p><strong>Min Speed: </strong>{stats.min_speed.toFixed(2)} km/h</p>
        <p><strong>Total Vehicles: </strong>{stats.total_count}</p>
      </div>

      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        <Bar data={barData} options={barOptions} />
      </div>
    </div>
  );
}