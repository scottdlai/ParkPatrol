import './App.css';

import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  // in [0, 1]
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImagePreview(URL.createObjectURL(selectedFile));
      setPrediction(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      // TODO replace api path with actual url
      const response = await fetch('localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setPrediction(data.occupiedProbability);
    } catch (err) {
      console.error('Prediction failed', err);
      setPrediction('Error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen w-screen bg-zinc-50">
      <div className="align-super w-md h-80 mx-auto p-4 border rounded-xl shadow space-y-4 bg-white">
        <h1 className="text-xl font-bold text-center">Paw Patrol</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />

          <label
            htmlFor="file-upload"
            className="block w-full px-4 py-2 text-center bg-gray-100 border border-gray-300 rounded cursor-pointer hover:bg-gray-200 font-medium"
          >
            {file ? 'Change Image' : 'Upload Image'}
          </label>

          {imagePreview && (
            <img
              src={imagePreview}
              alt="Selected"
              className="w-full h-auto rounded shadow"
            />
          )}

          <button
            type="submit"
            disabled={loading || !file}
            className="w-full px-4 py-2 bg-blue-600 text-white font-semibold rounded hover:bg-blue-700"
          >
            {loading ? 'Analyzing...' : 'Submit'}
          </button>
        </form>

        {prediction !== null && (
          <div className="text-center text-lg">
            Probability occupied:{' '}
            <strong>{(prediction * 100).toFixed(1)}%</strong>
          </div>
        )}
      </div>
    </div>
  );
}
