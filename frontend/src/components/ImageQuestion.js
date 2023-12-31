import React, { useState } from 'react';
import axios from 'axios';
import JSZip from 'jszip';
import './ImageQuestion.css'

const ImageQuestion = () => {
    const [imageUrls, setImageUrls] = useState([]);
    const [answers, setAnswers] = useState([]);
    const [responses, setResponses] = useState([]);
    const [currentIndex, setIndex] = useState([]);

    const fetchImages = async () => {
        try {
            const response = await axios.get('/get-images', { responseType: 'blob' });

            const zipFile = new Blob([response.data], { type: 'application/zip' });
            const zipUrl = URL.createObjectURL(zipFile);

            const zip = new JSZip();
            await zip.loadAsync(zipFile);
            const imageUrls = await Promise.all(
                Object.values(zip.files)
                    .filter(zipEntry => !zipEntry.dir)
                    .map(async (zipEntry) => {
                        const fileBlob = await zipEntry.async('blob');
                        return URL.createObjectURL(fileBlob);
                    })
            );

            setImageUrls(imageUrls);
            setResponses([]);
            setAnswers([])
            setIndex(0)
        } catch (error) {
            console.error('Error fetching images:', error);
        }
    };

    const submitAnswers = async () => {
        try {
            const response = await axios.post('/submit-answers', { answers });
            setResponses(response.data.responses);
        } catch (error) {
            console.error(error);
        }
    };

    const nextImage = () => {
        const newCurrentIndex = currentIndex + 1;
        if (newCurrentIndex < imageUrls.length) {
            setIndex(newCurrentIndex);
        }
    }

    const prevImage = () => {
        const newCurrentIndex = currentIndex - 1;
        if (newCurrentIndex >= 0) {
            setIndex(newCurrentIndex);
        }
    }

    return (
        <div>
            <button className='load-button' onClick={fetchImages}>Load Images</button>
            <div>
                <img src={imageUrls[currentIndex]} alt={`Image ${currentIndex + 1}`} className="image-content" />
                <br />
                <div className='input-div'>
                    <input className='input-input'
                        type="text"
                        value={answers[currentIndex] || ''}
                        onChange={(e) => {
                            const newAnswers = [...answers];
                            newAnswers[currentIndex] = e.target.value
                            setAnswers(newAnswers)
                        }}
                    />
                    <button onClick={prevImage}>Prev</button>
                    <button onClick={nextImage}>Next</button>
                </div>
            </div>
            <button onClick={submitAnswers}>Submit</button>
            {responses && <p>{responses}</p>}
        </div>
    );
};

export default ImageQuestion
