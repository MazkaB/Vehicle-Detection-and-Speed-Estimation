import React, { useEffect, useState } from 'react';
import Hls from 'hls.js';
import { useParams } from 'react-router-dom';
import axios from 'axios';

function VideoStream() {
  const { videoId } = useParams();
  const [playlistUrl, setPlaylistUrl] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const startStream = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/videos/${videoId}/start_stream`);
        const fullPath = `http://localhost:5000${data.m3u8_url}`;
        setPlaylistUrl(fullPath);
      } catch (err) {
        console.error('Error starting HLS stream:', err);
        setError('Failed to load HLS stream');
      }
    };
    startStream();
  }, [videoId]);

  useEffect(() => {
    if (playlistUrl && Hls.isSupported()) {
      const video = document.getElementById('video-player');
      const hls = new Hls();

      hls.loadSource(playlistUrl);
      hls.attachMedia(video);

      hls.on(Hls.Events.ERROR, (event, data) => {
        console.error('HLS error:', data);
      });
    } else if (playlistUrl) {
      const video = document.getElementById('video-player');
      video.src = playlistUrl;
    }
  }, [playlistUrl]);

  return (
    <div>
      <h2>Video Stream for ID: {videoId}</h2>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <video
        id="video-player"
        width="640"
        height="360"
        controls
        autoPlay
        style={{ backgroundColor: 'black' }}
      />

      {playlistUrl && (
        <p>
          Now playing HLS from: <code>{playlistUrl}</code>
        </p>
      )}
    </div>
  );
}

export default VideoStream;