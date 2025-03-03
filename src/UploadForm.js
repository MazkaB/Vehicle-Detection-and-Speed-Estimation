import React, { useState } from 'react';
import { uploadVideo } from './api';

export default function UploadForm({ onUploadSuccess }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [file, setFile] = useState(null);
  const [hlsLink, setHlsLink] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', name);
    formData.append('description', description);
    if (file) {
      formData.append('file', file);
    } else if (hlsLink) {
      formData.append('hls_link', hlsLink);
    } else {
      alert("Please provide either a file or an HLS link!");
      return;
    }
    await uploadVideo(formData);
    setName('');
    setDescription('');
    setFile(null);
    setHlsLink('');
    if (onUploadSuccess) {
      onUploadSuccess();
    }
  }

  return (
    <div className="card mb-4">
      <div className="card-header">Add New Video / Stream</div>
      <div className="card-body">
        <form onSubmit={handleSubmit} className="row g-3">
          <div className="col-md-4">
            <label className="form-label">Name</label>
            <input
              className="form-control"
              value={name}
              onChange={e=>setName(e.target.value)}
              required
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Description</label>
            <input
              className="form-control"
              value={description}
              onChange={e=>setDescription(e.target.value)}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Video File</label>
            <input
              className="form-control"
              type="file"
              onChange={e=>setFile(e.target.files[0])}
            />
          </div>
          <div className="col-md-4">
            <label className="form-label">Or HLS Link</label>
            <input
              className="form-control"
              value={hlsLink}
              onChange={e=>setHlsLink(e.target.value)}
              placeholder="https://example.com/path.m3u8"
            />
          </div>
          <div className="col-md-4 d-flex align-items-end">
            <button type="submit" className="btn btn-primary">Upload</button>
          </div>
        </form>
      </div>
    </div>
  );
}
