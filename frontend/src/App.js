import FileUpload from './components/FileUpload';
import './App.css';
import FileDownload from './components/FileDownlad';
import VAEPrepare from './components/VAEPrepare';
import ImageQuestion from './components/ImageQuestion';
import QuestionHeading from './components/QuestionHeading';
import { help_upload, help_vae, help_manual, help_download } from './mocks/helps';

function App() {
  return (
    <div className="App">
      <div className='card'>
        <QuestionHeading name="1. File Upload" help={help_upload} />
        <FileUpload />

        <QuestionHeading name="2. VAE Preparation" help={help_vae} />
        <VAEPrepare />

        <QuestionHeading name="3. Manual Data Annotation" help={help_manual} />
        <ImageQuestion />

        <QuestionHeading name="4. Download Files" help={help_download} />
        <FileDownload />
      </div>
    </div>
  );
}

export default App;
