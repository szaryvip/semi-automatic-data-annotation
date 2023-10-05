import FileUpload from './components/FileUpload';
import './App.css';
import FileDownload from './components/FileDownlad';
import VAEPrepare from './components/VAEPrepare';
import ImageQuestion from './components/ImageQuestion';

function App() {
  return (
    <div className="App">
      <h1>File Upload</h1>
      <FileUpload />

      <h1>VAE Preparation</h1>
      <VAEPrepare />

      <h1>Manual Data Annotation</h1>
      <ImageQuestion />

      <h1>Download Files</h1>
      <FileDownload />
    </div>
  );
}

export default App;
