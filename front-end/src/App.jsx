import './App.css';

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
            <div
                className="card w-full max-w-2xl p-6 rounded-2xl shadow-xl space-y-6"
                style={{ backgroundColor: '#3F5371' }}
            >
                {/* Logo + Title horizontally */}
                <div className="flex items-center space-x-4 justify-center">
                    <div className="w-22 h-22 rounded-full overflow-hidden">
                        <img
                            src="/assets/Picture1.png"
                            alt="Logo"
                            className="object-cover w-full h-full"
                        />
                    </div>
                    <div>
                        <h1
                            className="text-5xl font-extrabold text-primary"
                            style={{ color: '#FFE8BF' }}
                        >
                            Park Patrol
                        </h1>
                        <p className="text-lg text-white text-center">
                            Check stall occupancy with ML
                        </p>
                    </div>
                </div>

                {/* Sticky File Input and Button */}
                <div
                    className="sticky top-0  z-10 pb-2 pt-1"
                    style={{ backgroundColor: '#3F5371' }}
                >
                    <fieldset className="fieldset">
                        <legend className="fieldset-legend text-grey-300 ">
                            Upload a bird's-eye view photo with visible parking
                            stall lines
                        </legend>
                        <form
                            className="flex items-center gap-3 w-full"
                            onSubmit={handleSubmit}
                        >
                            <input
                                disabled={loading}
                                id="file-upload"
                                type="file"
                                accept="image/*"
                                onChange={handleFileChange}
                                className="file-input file-input-bordered file-input-sm w-full"
                            />

                            <button
                                type="submit"
                                className="btn btn-sm disabled:opacity-50 disabled:cursor-not-allowed text-black"
                                style={{ backgroundColor: '#FFE8BF' }}
                                disabled={loading}
                            >
                                {!loading ? 'Find lot status' : 'Analyzing...'}
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
                {spots !== null &&
                    (() => {
                        // Calculate all values upfront
                        const totalSpots = spots;
                        const vacantCount = vacants ?? 0;
                        const occupiedCount = occupieds ?? 0;
                        const vacancyRate =
                            totalSpots > 0 ? vacantCount / totalSpots : 0;

                        // Determine status once
                        let status, statusClass, statusImage, statusText;

                        if (vacancyRate <= 0.15) {
                            status = 'critical';
                            statusClass = 'text-red-400';
                            statusImage = '/assets/sad_rat.jpg';
                            statusText = 'Almost or already full!';
                        } else if (vacancyRate <= 0.3) {
                            status = 'limited';
                            statusClass = 'text-amber-400';
                            statusImage = '/assets/unsure_rat.jpg';
                            statusText = 'Limited spaces';
                        } else if (vacancyRate > 0.5) {
                            status = 'plenty';
                            statusClass = 'text-green-400';
                            statusImage = '/assets/smirk_rat.jpg';
                            statusText = 'Plenty available!';
                        } else {
                            status = 'moderate';
                            statusClass = 'text-blue-400';
                            statusImage = '/assets/happy_rat.jpg';
                            statusText = 'Moderate spaces';
                        }

                        return (
                            <div className="flex items-stretch gap-4 overflow-x-auto rounded-box border border-base-300 bg-base-100 p-4">
                                <table className="table table-sm w-full max-w-[60%]">
                                    <thead>
                                        <tr
                                            className="text-base-content"
                                            style={{ color: '#FFE8BF' }}
                                        >
                                            <th>Category</th>
                                            <th>Count</th>
                                            <th>Percentage</th>
                                        </tr>
                                    </thead>
                                    <tbody className="text-white">
                                        <tr>
                                            <td className="text-green-500 italic">
                                                Vacant
                                            </td>
                                            <td>
                                                {vacantCount}/{totalSpots}
                                            </td>
                                            <td>
                                                {totalSpots > 0
                                                    ? `${(vacancyRate * 100).toFixed(1)}%`
                                                    : '—'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td className="text-red-500 italic">
                                                Occupied
                                            </td>
                                            <td>
                                                {occupiedCount}/{totalSpots}
                                            </td>
                                            <td>
                                                {totalSpots > 0
                                                    ? `${((occupiedCount / totalSpots) * 100).toFixed(1)}%`
                                                    : '—'}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>

                                <div className="border-l border-gray-300"></div>

                                <div className="flex flex-col justify-center items-center w-24 min-h-full px-2">
                                    <div className="flex flex-col items-center justify-center gap-1 h-full">
                                        <div className="avatar">
                                            <div className="w-16 rounded-full">
                                                <img
                                                    src={statusImage}
                                                    className="object-cover"
                                                    alt={`Parking status: ${status}`}
                                                />
                                            </div>
                                        </div>
                                        <span
                                            className={`text-sm text-center w-full ${statusClass}`}
                                        >
                                            {statusText}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        );
                    })()}
            </div>
        </div>
    );
}
