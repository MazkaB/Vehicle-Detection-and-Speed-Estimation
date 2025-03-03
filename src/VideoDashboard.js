import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchVideos, deleteVideo } from './api'; 
import UploadForm from './UploadForm';

export default function VideoDashboard() {
  const [videos, setVideos] = useState([]);

  async function loadVideos() {
    const data = await fetchVideos();
    setVideos(data);
  }

  useEffect(() => {
    loadVideos();
  }, []);

  
  async function handleDelete(videoId) {
    if (window.confirm("Are you sure you want to delete this video?")) {
      try {
        await deleteVideo(videoId);
        loadVideos();
      } catch (err) {
        console.error("Failed to delete video:", err);
        alert("Error deleting the video.");
      }
    }
  }

  return (
    <div className="container">
      <UploadForm onUploadSuccess={loadVideos} />

      <div className="row">
        {videos.map((video) => (
          <div className="col-md-4 mb-3" key={video.id}>
            <div className="card">
              <div className="card-body">
                <h5 className="card-title">{video.name}</h5>
                <p className="card-text">{video.description}</p>
                <Link to={`/videos/${video.id}`} className="btn btn-sm btn-success me-2">
                  View Live
                </Link>
                <Link to={`/videos/${video.id}/stats`} className="btn btn-sm btn-warning me-2">
                  Statistics
                </Link>
                {/* NEW OR MODIFIED: Delete button */}
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => handleDelete(video.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}