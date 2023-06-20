import FileUpload from './components/FileUpload';
import './App.css';
import DataClustering from './components/DataClustering';
import FileDownload from './components/FileDownlad';
import ImageQuestion from './components/ImageQuestion';

function App() {
  return (
    <div className="App">
      <h1>File Upload</h1>
      <FileUpload />

      <h1>Data Clustering</h1>
      <DataClustering />

      <h1>Manual Data Annotation</h1>
      <ImageQuestion />

      <h1>Download Files</h1>
      <FileDownload />
    </div>
  );
}

export default App;
