import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Dropzone from 'react-dropzone';
import uploadIcon from '../assets/upload.png'
import './FileUpload.css'

const FileUpload = () => {
    const [files, setFiles] = useState([]);
    const [errorMessage, setErrorMessage] = useState('');

    const handleDrop = (acceptedFiles) => {
        setFiles(acceptedFiles);
        setErrorMessage('');
    };

    const validateFileExtension = (fileName) => {
        const acceptedExtensions = ['.jpg', '.jpeg', '.png'];
        const fileExtension = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
        return acceptedExtensions.includes(fileExtension);
    };

    const handleUpload = async () => {
        const formData = new FormData();
        const invalidFiles = [];

        files.forEach((file) => {
            if (validateFileExtension(file.name)) {
                formData.append('files', file);
            } else {
                invalidFiles.push(file.name);
            }
        });

        if (invalidFiles.length > 0) {
            setErrorMessage(`Invalid file extensions: ${invalidFiles.join(', ')}`);
            return;
        }

        const headers = {
            'Content-Type': 'multipart/form-data',
        };

        try {
            const response = await axios.post('/upload', formData, { headers });
            console.log(response);
        } catch (error) {
            console.error(error);
            setErrorMessage('Error uploading files');
        }
    };

    const removePrevious = async () => {
        try {
            const response = await axios.delete('/delete-files');
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className='upload-div'>
            <Dropzone onDrop={handleDrop} multiple>
                {({ getRootProps, getInputProps }) => (
                    <div className='dropbox-div' {...getRootProps()}>
                        <input {...getInputProps()} />
                        <img src={uploadIcon} className='upload-image' />
                        <p className='prompt'>Drag and drop files here or click to select files</p>
                    </div>
                )}
            </Dropzone>
            {errorMessage && <p className='error-message'>{errorMessage}</p>}
            <div className='data-buttons'>
                <button onClick={removePrevious}>Remove Previous</button>
                <button onClick={handleUpload}>Upload</button>
            </div>
        </div>
    );
};

export default FileUpload;
