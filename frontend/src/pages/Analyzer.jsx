import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, AlertCircle } from 'lucide-react';
import { analyzeAccount } from '../utils/api';

export default function Analyzer() {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) {
      setError('Please enter a username');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await analyzeAccount(username);
      navigate(`/result/${username}`, { state: { data } });
    } catch (err) {
      setError(err.error || 'Failed to analyze account. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 to-white">
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        {/* Header */}
        <div className="text-center mb-12 fade-in">
          <h1 className="text-4xl sm:text-5xl font-bold text-dark mb-4">
            Analyze Instagram <span className="gradient-primary bg-clip-text text-transparent">Account</span>
          </h1>
          <p className="text-lg text-gray-600">
            Enter any public Instagram username to detect if it's real or fake
          </p>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 sm:p-12 fade-in">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Error Alert */}
            {error && (
              <div className="flex gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="text-danger flex-shrink-0 w-5 h-5 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-danger">Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            )}

            {/* Username Input */}
            <div>
              <label htmlFor="username" className="block text-sm font-semibold text-dark mb-2">
                Instagram Username
              </label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 text-lg">@</span>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="john_doe"
                  className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-primary transition"
                  disabled={loading}
                />
              </div>
              <p className="mt-2 text-sm text-gray-500">
                Enter the username without @ symbol. Example: cristiano
              </p>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 gradient-primary text-white rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Analyze Account
                </>
              )}
            </button>
          </form>

          {/* Info Box */}
          <div className="mt-8 p-4 bg-blue-50 border-l-4 border-primary rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>💡 Tip:</strong> Make sure the account is public. The system analyzes public profile data to determine authenticity.
            </p>
          </div>
        </div>

        {/* Features Highlight */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl mb-2">🚀</div>
            <h3 className="font-semibold text-dark mb-2">Instant Analysis</h3>
            <p className="text-sm text-gray-600">Get results in seconds</p>
          </div>
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl mb-2">📊</div>
            <h3 className="font-semibold text-dark mb-2">Detailed Insights</h3>
            <p className="text-sm text-gray-600">Understand the reasons behind the result</p>
          </div>
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl mb-2">🎯</div>
            <h3 className="font-semibold text-dark mb-2">High Accuracy</h3>
            <p className="text-sm text-gray-600">AI-powered detection system</p>
          </div>
        </div>
      </div>
    </div>
  );
}
