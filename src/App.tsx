

import './styles/App.scss'
import Footer from './components/Footer';
import Header from './components/Header';
import MainPage from './components/MainPage';


const App: React.FC = () => {
  // const [userId, setUserId] = useState<string | null>(null);
  
  return (
    <>
      <Header/>
      <MainPage/>
      <Footer/>
    </>
  );
};

export default App;
