import axios from 'axios';

const API_BASE = "http://localhost:5000/api";

export async function fetchVideos() {
  const res = await axios.get(`${API_BASE}/videos`);
  return res.data;
}

export async function uploadVideo(formData) {
  const res = await axios.post(`${API_BASE}/videos`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

export async function fetchVideoStats(videoId) {
  const res = await axios.get(`${API_BASE}/videos/${videoId}/stats`);
  return res.data;
}

export async function deleteVideo(videoId) {
    const res = await axios.delete(`${API_BASE}/videos/${videoId}`);
    return res.data;  // e.g. {message: "Video ID=xx deleted"}
  }
  