import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Analyzer from './pages/Analyzer';
import ResultPage from './pages/ResultPage';
import AnalysisPage from './pages/AnalysisPage';

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyzer" element={<Analyzer />} />
        <Route path="/result/:username" element={<ResultPage />} />
        <Route path="/analysis/:username" element={<AnalysisPage />} />
      </Routes>
      <Footer />
    </Router>
  );
}
