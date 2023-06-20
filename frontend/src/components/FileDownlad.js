import React from 'react';
import axios from 'axios';

const FileDownload = () => {
    const handleDownload = async () => {
        try {
            const response = await axios.get('/download_files', { responseType: 'blob' });

            const downloadUrl = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.setAttribute('download', 'data.zip');
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div>
            <button onClick={handleDownload}>Download Files</button>
        </div>
    );
};

export default FileDownload;
