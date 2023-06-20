import React, { useState } from 'react';
import axios from 'axios';

const DataClustering = () => {
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const handleButtonClick = async () => {
        try {
            const response = await axios.get('/cluster_data');
            setMessage(response.data.message);
            setError('');
        } catch (error) {
            setMessage('');
            setError('Error occurred. Please try again.');
        }
    };

    return (
        <div>
            <button onClick={handleButtonClick}>Proceed</button>
            {message && <p>{message}</p>}
            {error && <p>{error}</p>}
        </div>
    );
};

export default DataClustering;
