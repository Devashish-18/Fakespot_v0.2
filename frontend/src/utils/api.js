import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const analyzeAccount = async (username) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analyze`, {
      params: { username: username.trim() }
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || { error: 'Failed to analyze account' };
  }
};

export const exportJSON = (data, filename = 'analysis_report.json') => {
  const dataStr = JSON.stringify(data, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
