import React, { useEffect, useState } from 'react';
import { useParams, useLocation, Link, useNavigate } from 'react-router-dom';
import { Download, ChevronRight } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import ReasonBox from '../components/ReasonBox';
import { exportJSON } from '../utils/api';
import { formatCountSafe } from '../utils/countFormatter';

export default function ResultPage() {
  const { username } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!location.state?.data) {
      navigate('/analyzer');
      return;
    }
    setData(location.state.data);
  }, [location, navigate]);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="spinner" style={{ width: '50px', height: '50px' }}></div>
      </div>
    );
  }

  const isPredictionFake = data.prediction === 1 || data.prediction === 'FAKE';
  const predictionClass = isPredictionFake ? 'gradient-danger' : 'gradient-success';
  const predictionText = isPredictionFake
    ? '⚠️ This account displays multiple indicators of artificial activity, bot-like behavior, or fraudulent patterns based on analyzed signals.'
    : '✓ This account demonstrates authentic behavior patterns consistent with genuine human activity and engagement.';

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 to-white py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="text-center mb-12 fade-in">
          <h1 className="text-4xl font-bold text-dark mb-2">Account Analysis Complete</h1>
          <p className="text-xl text-gray-600 mb-4">@{data.username}</p>
          <p className="text-gray-500">Real-time analysis based on profile signals and behavioral patterns</p>
        </div>

        {/* Prediction Card */}
        <div className={`${predictionClass} rounded-2xl p-8 sm:p-12 text-white text-center mb-12 slide-up shadow-xl`}>
          <div className="text-lg font-semibold mb-3 opacity-90">Prediction Result</div>
          <div className="text-6xl sm:text-7xl font-bold mb-4">
            {isPredictionFake ? '🚨 FAKE' : '✓ REAL'}
          </div>
          <div className="text-2xl font-semibold mb-6">
            {Math.round(data.confidence * 100)}% Confidence
          </div>
          <div className="text-lg opacity-90 max-w-3xl mx-auto leading-relaxed">
            {predictionText}
          </div>
          <div className="mt-8 pt-8 border-t border-white border-opacity-20">
            <p className="text-sm opacity-75">Scroll down to see detailed metrics, signals, and interactive performance graphs</p>
          </div>
        </div>

        {/* Profile Metrics Grid */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-dark mb-6">Profile Metrics</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard 
              label="Followers" 
              value={formatCountSafe(data.profile_data.followers)}
              color="primary"
              formatAsCount={true}
            />
            <MetricCard 
              label="Following" 
              value={formatCountSafe(data.profile_data.following)}
              color="secondary"
              formatAsCount={true}
            />
            <MetricCard 
              label="Posts" 
              value={formatCountSafe(data.profile_data.posts)}
              color="success"
              formatAsCount={true}
            />
            <MetricCard 
              label="Engagement Rate" 
              value={(data.profile_data.engagement_rate * 100).toFixed(2)}
              unit="%"
              color="warning"
            />
            <MetricCard 
              label="Account Age" 
              value={data.profile_data.account_age_days}
              unit="days"
              color="primary"
            />
            <MetricCard 
              label="Bio Length" 
              value={data.profile_data.bio_length}
              unit="chars"
              color="secondary"
            />
            <MetricCard 
              label="Profile Picture" 
              value={data.profile_data.has_profile_pic ? 'Yes' : 'No'}
              color={data.profile_data.has_profile_pic ? 'success' : 'danger'}
            />
            <MetricCard 
              label="Private Account" 
              value={data.profile_data.is_private ? 'Yes' : 'No'}
              color={data.profile_data.is_private ? 'warning' : 'success'}
            />
          </div>
        </div>

        {/* Why This Result Section */}
        <div className="mb-12">
          <div className="bg-white rounded-2xl p-8 border-2 border-primary mb-8">
            <h2 className="text-2xl font-bold text-dark mb-4">🔍 Key Signals Analyzed</h2>
            <p className="text-gray-600 mb-6">
              The following signals were evaluated to determine if this account is real or fake:
            </p>
            <div className="space-y-4">
              {data.reasons && data.reasons.length > 0 ? (
                data.reasons.map((reason, idx) => (
                  <ReasonBox key={idx} reason={reason} />
                ))
              ) : (
                <p className="text-gray-500 italic">No specific signals to highlight - general account patterns assessed.</p>
              )}
            </div>
          </div>
        </div>

        {/* Next Steps Section */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 mb-12 border-l-4 border-primary">
          <h2 className="text-2xl font-bold text-dark mb-4">📊 Next Steps</h2>
          <p className="text-gray-700 mb-6">
            View detailed performance analysis with interactive graphs that break down each metric and explain how it impacts the final result:
          </p>
          <ul className="space-y-2 text-gray-600 mb-6">
            <li>✓ <strong>4 Interactive Charts</strong> - Bar, Radar, Line, and Doughnut charts showing actual data</li>
            <li>✓ <strong>Detailed Explanations</strong> - Each graph tells you WHY signals indicate fake or real</li>
            <li>✓ <strong>Metric Breakdown</strong> - Comprehensive table of all 8+ analyzed metrics</li>
            <li>✓ <strong>Export Reports</strong> - Download analysis in JSON format</li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 mb-12">
          <button
            onClick={() => exportJSON(data, `${username}_analysis.json`)}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-white border-2 border-primary text-primary rounded-lg font-semibold hover:bg-primary hover:text-white transition"
          >
            <Download size={20} />
            Export JSON Report
          </button>
          <Link
            to={`/analysis/${username}`}
            state={{ data }}
            className="flex items-center justify-center gap-2 px-6 py-3 gradient-primary text-white rounded-lg font-semibold hover:shadow-lg transition flex-1"
          >
            🎯 View Performance Graphs & Analysis
            <ChevronRight size={20} />
          </Link>
        </div>

        {/* Analyze Another */}
        <div className="text-center p-8 bg-white rounded-xl">
          <h3 className="text-xl font-semibold text-dark mb-3">Want to check another account?</h3>
          <Link to="/analyzer" className="text-primary font-semibold hover:underline">
            Analyze another account →
          </Link>
        </div>
      </div>
    </div>
  );
}
