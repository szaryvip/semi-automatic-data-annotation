import React, { useState } from 'react';
import axios from 'axios';

const ImageQuestion = () => {
    const [imageUrl, setImageUrl] = useState('');
    const [answer, setAnswer] = useState('');
    const [response, setResponse] = useState('');

    const fetchImage = () => {
        fetch('get_image/')
            .then((response) => response.blob())
            .then((blob) => {
                const imageUrl = URL.createObjectURL(blob);
                setImageUrl(imageUrl);
            })
            .catch((error) => {
                console.error('Error fetching image:', error);
            });
    };

    const submitAnswer = async () => {
        try {
            const response = await axios.post('submit_answer/', { "answer": answer });
            setResponse(response.data.message);
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div>
            {imageUrl && <img src={imageUrl} alt="Image" />}
            <p></p>
            <input type="text" value={answer} onChange={(e) => setAnswer(e.target.value)} />
            <button onClick={submitAnswer}>Submit</button>
            {response && <p>{response}</p>}
            <p></p>
            <button onClick={fetchImage}>Next Question</button>
        </div>
    );
};

export default ImageQuestion;
