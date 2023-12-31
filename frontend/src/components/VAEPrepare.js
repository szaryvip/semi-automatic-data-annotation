import React, { useState } from 'react';
import axios from 'axios';
import { ThreeDots } from 'react-loader-spinner';
import './VAEPrepare.css'

const VAEPrepare = () => {
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const prepareVAE = async () => {
        try {
            setLoading(true);
            const response = await axios.get('/prepare-vae');
            setMessage(response.data);
        } catch (error) {
            setLoading(false);
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <button onClick={prepareVAE}>Prepare VAE model</button>
            {loading ? (
                <div className='waiting-div'>
                    <p>Sit tight, it could take a while...</p>
                    <ThreeDots color="#5300b3" height={50} width={50} />
                </div>
            ) : (
                message && <p>{message}</p>
            )}
        </div>
    );
};

export default VAEPrepare;
