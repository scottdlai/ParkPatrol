import './App.css';
import Picture1 from '../public/assets/Picture1.png';

import React, { useState } from 'react';

const API_URL = 'http://localhost:8000/api/predict';

export default function App() {
    const [file, setFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    // in [0, 1]
    // const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const [spots, setSpots] = useState(null);
    const [vacants, setVacants] = useState(null);
    const [occupieds, setOccupieds] = useState(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setImagePreview(URL.createObjectURL(selectedFile));
            // setPrediction(null);
            setSpots(null);
            setVacants(null);
            setOccupieds(null);
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
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            console.log(data);

            setImagePreview(data.img);
            setSpots(data.stats?.spots);
            setVacants(data.stats?.vacants);
            setOccupieds(data.stats?.occupieds);
        } catch (err) {
            console.error('Prediction failed', err);
            setSpots(null);
            setVacants(null);
            setOccupieds(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex justify-center items-start min-h-screen w-screen bg-base-200 py-10 px-4 text-[#3F5371]">
            <div className="card w-full max-w-2xl p-6 bg-white rounded-2xl shadow-xl space-y-6">

                {/* Logo + Title horizontally */}
                <div className="flex items-center space-x-4 justify-center">
                    <div className="w-22 h-22 rounded-full overflow-hidden">
                        <img src="/assets/Picture1.png" alt="Logo" className="object-cover w-full h-full" />
                    </div>
                    <div>
                        <h1 className="text-5xl font-extrabold text-primary">Park Patrol</h1>
                        <p className="text-lg text-slate-500 text-center">Check stall occupancy with ML</p>
                    </div>
                </div>

                {/* Sticky File Input and Button */}
                <div className="sticky top-0 bg-white z-10 pb-2 pt-1">
                    <fieldset className="fieldset">
                        <legend className="fieldset-legend text-slate-500 ">Upload a bird's-eye view photo with visible parking stall lines</legend>
                    <form className="flex items-center gap-3 w-full" onSubmit={handleSubmit}>
                        <input
                            id="file-upload"
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            className="file-input file-input-bordered file-input-sm w-full"
                        />

                        <button
                            type="submit"
                            className="btn btn-sm btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Find lot status
                        </button>
                    </form>
                    </fieldset>
                </div>

                {/* Image preview section */}
                {imagePreview && (
                    <div className="bg-base-100 p-2 rounded-lg border shadow-md max-h-[60svh] overflow-auto flex justify-center">
                        <img
                            src={imagePreview}
                            alt="Selected"
                            className="rounded object-contain max-h-full"
                        />
                    </div>
                )}

                {/* Table + icon */}
                {spots !== null && (
                    <div className="flex items-stretch gap-4 overflow-x-auto rounded-box border border-base-300 bg-base-100 p-4">
                        <table className="table table-sm table-zebra w-full max-w-[60%]">
                            <thead>
                            <tr className="text-base-content">
                                <th>Category</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                            </thead>
                            <tbody>
                            {vacants !== null && (
                                <tr>
                                    <td>Vacant</td>
                                    <td>{vacants}/{spots}</td>
                                    <td>{spots > 0 ? `${((vacants / spots) * 100).toFixed(1)}%` : '—'}</td>
                                </tr>
                            )}
                            {occupieds !== null && (
                                <tr>
                                    <td>Occupied</td>
                                    <td>{occupieds}/{spots}</td>
                                    <td>{spots > 0 ? `${((occupieds / spots) * 100).toFixed(1)}%` : '—'}</td>
                                </tr>
                            )}
                            </tbody>
                        </table>

                        {/* Divider */}
                        <div className="border-l border-gray-300"></div>

                        {/* Centered caption container */}
                        {/* Avatar + Text Container - Fixed Centering */}
                        <div className="flex flex-col justify-center items-center w-24 min-h-full px-2">
                            <div className="flex flex-col items-center justify-center gap-1 h-full">
                                <div className="avatar">
                                    <div className="w-16 rounded-full">
                                        {vacants === 0 ? (
                                            <img
                                                src="https://img.daisyui.com/images/profile/demo/yellingcat@192.webp"
                                                className="object-cover"
                                                alt="Full parking"
                                            />
                                        ) : occupieds === 0 ? (
                                            <img
                                                src="https://img.daisyui.com/images/profile/demo/spiderperson@192.webp"
                                                className="object-cover"
                                                alt="Empty parking"
                                            />
                                        ) : (vacants / spots) > 0.5 ? (
                                            <img
                                                src="https://img.daisyui.com/images/profile/demo/yellingwoman@192.webp"
                                                className="object-cover"
                                                alt="Many spaces"
                                            />
                                        ) : (
                                            <img
                                                src="https://img.daisyui.com/images/profile/demo/batperson@192.webp"
                                                className="object-cover"
                                                alt="Few spaces"
                                            />
                                        )}
                                    </div>
                                </div>
                                <span className="text-sm font-medium text-center w-full">
          {(vacants / spots) <= 0.1 ? (
              "Almost or already full!"
          ) : (vacants / spots) <= 0.3 ? (
              "Limited spaces"
          ) : (vacants / spots) > 0.5 ? (
              "Plenty of spaces!"
          ) : (
              "Moderate availability"
          )}
        </span>
                            </div>
                        </div>

                    </div>
                )}

            </div>
        </div>
    );
}
