import React, { useEffect, useState } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import {
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import ChartCard from '../components/ChartCard';
import { generateChartExplanations } from '../utils/explanations';
import { formatCountSafe } from '../utils/countFormatter';

export default function AnalysisPage() {
  const { username } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [explanations, setExplanations] = useState({});

  // Custom tooltip formatter for charts
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      const isLargeNumber = value > 1000;
      const displayValue = isLargeNumber ? formatCountSafe(value) : value;
      return (
        <div style={{
          backgroundColor: '#fff',
          border: '1px solid #ccc',
          borderRadius: '8px',
          padding: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <p style={{ margin: 0, color: '#333' }}>
            <strong>{payload[0].name}:</strong> {displayValue}
          </p>
        </div>
      );
    }
    return null;
  };

  useEffect(() => {
    if (!location.state?.data) {
      navigate('/analyzer');
      return;
    }
    const analysisData = location.state.data;
    setData(analysisData);
    setExplanations(generateChartExplanations(analysisData));
  }, [location, navigate]);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="spinner" style={{ width: '50px', height: '50px' }}></div>
      </div>
    );
  }

  // Prepare data for charts
  const barChartData = [
    { metric: 'Followers', value: Math.min(data.profile_data.followers, 5000) },
    { metric: 'Following', value: Math.min(data.profile_data.following, 5000) },
    { metric: 'Posts', value: data.profile_data.posts * 100 }
  ];

  const radarData = [
    { feature: 'Engagement', value: Math.min(data.profile_data.engagement_rate * 100, 100) },
    { feature: 'Account Age', value: Math.min(data.profile_data.account_age_days * 2, 100) },
    { feature: 'Profile Completeness', value: (data.profile_data.has_profile_pic && data.profile_data.bio_length > 0 ? 100 : 30) },
    { feature: 'Post Activity', value: Math.min(data.profile_data.posts * 10, 100) },
    { feature: 'Network Quality', value: (data.profile_data.following > 0 ? Math.min((data.profile_data.followers / data.profile_data.following) * 100, 100) : 50) }
  ];

  const lineChartData = Array.from({ length: Math.max(data.profile_data.posts, 5) }, (_, i) => ({
    day: `Day ${i + 1}`,
    likes: Math.round(data.profile_data.avg_likes * (1 + Math.random() * 0.5))
  }));

  const suspiciousScore = data.confidence * 100;
  const doughnutData = [
    { name: 'Profile Signals', value: Math.round(suspiciousScore * 0.25) },
    { name: 'Network Signals', value: Math.round(suspiciousScore * 0.45) },
    { name: 'Engagement Signals', value: Math.round(suspiciousScore * 0.2) },
    { name: 'Authenticity', value: Math.round(100 - suspiciousScore) }
  ];

  const COLORS = ['#ef4444', '#f59e0b', '#ec4899', '#10b981'];

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 to-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="text-center mb-12 fade-in">
          <h1 className="text-4xl font-bold text-dark mb-2">Detailed Performance Analysis</h1>
          <p className="text-gray-600 text-lg">@{data.username} - In-depth account metrics and interactive visualization</p>
        </div>

        {/* Prediction Summary Box */}
        <div className={`${data.prediction === 1 || data.prediction === 'FAKE' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'} rounded-2xl p-8 border-2 mb-12 fade-in`}>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="md:col-span-2">
              <h2 className="text-2xl font-bold text-dark mb-3">
                {data.prediction === 1 || data.prediction === 'FAKE' ? '🚨 Account Classified as FAKE' : '✓ Account Classified as REAL'}
              </h2>
              <p className="text-gray-700 mb-4 leading-relaxed">
                {data.prediction === 1 || data.prediction === 'FAKE'
                  ? `Based on comprehensive analysis of ${data.profile_data.followers > 0 ? 'account metrics' : 'available data'}, this account displays multiple indicators of artificial activity. The patterns detected suggest the account may be using automated systems, purchasing followers, or engaging in other inauthentic behavior. Key concerns include the follower-to-following ratio, engagement patterns, profile completeness, and account age.`
                  : `This account demonstrates behavior patterns consistent with genuine human activity. The analyzed metrics show healthy engagement rates, realistic follower growth, complete profile information, and authentic interaction patterns. These indicators collectively suggest this is a legitimate Instagram account.`
                }
              </p>
              <p className="text-lg font-semibold">
                Confidence Level: <span className={data.prediction === 1 || data.prediction === 'FAKE' ? 'text-danger' : 'text-success'}>{Math.round(data.confidence * 100)}%</span>
              </p>
            </div>
            <div className={`${data.prediction === 1 || data.prediction === 'FAKE' ? 'bg-red-100' : 'bg-green-100'} rounded-lg p-6 flex flex-col justify-center items-center`}>
              <div className="text-5xl font-bold mb-2">{Math.round(data.confidence * 100)}%</div>
              <div className="text-center text-sm font-semibold text-gray-700">
                {data.prediction === 1 || data.prediction === 'FAKE' ? 'Likely FAKE' : 'Likely REAL'}
              </div>
            </div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Bar Chart */}
          <ChartCard
            title="Account Metrics Overview"
            explanation={explanations.bar}
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Radar Chart */}
          <ChartCard
            title="Account Health Indicators"
            explanation={explanations.radar}
          >
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <PolarAngleAxis dataKey="feature" />
                <PolarRadiusAxis />
                <Radar
                  name="Score"
                  dataKey="value"
                  stroke="#6366f1"
                  fill="#6366f1"
                  fillOpacity={0.6}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '8px'
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Line Chart */}
          <ChartCard
            title="Engagement Growth Trend"
            explanation={explanations.line}
          >
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lineChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="likes"
                  stroke="#ec4899"
                  strokeWidth={2}
                  dot={{ fill: '#ec4899', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Doughnut Chart */}
          <ChartCard
            title="Suspicious Score Breakdown"
            explanation={explanations.doughnut}
          >
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={doughnutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {doughnutData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Detailed Metrics Table */}
        <div className="mt-12 bg-white rounded-xl shadow p-8">
          <h2 className="text-2xl font-bold text-dark mb-6">Detailed Profile Analysis</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b-2 border-gray-200">
                <tr className="bg-gray-50">
                  <th className="px-4 py-3 text-left font-semibold text-dark">Metric</th>
                  <th className="px-4 py-3 text-right font-semibold text-dark">Value</th>
                  <th className="px-4 py-3 text-left font-semibold text-dark">Status</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Username</td>
                  <td className="px-4 py-3 text-right">@{data.username}</td>
                  <td className="px-4 py-3"><span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Provided</span></td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Followers</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.followers.toLocaleString()}</td>
                  <td className="px-4 py-3"><span className="text-gray-600">{data.profile_data.followers > 1000 ? '📈 Good' : '📉 Low'}</span></td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Following</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.following.toLocaleString()}</td>
                  <td className="px-4 py-3">
                    {data.profile_data.following > data.profile_data.followers * 3 ? (
                      <span className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded">⚠️ Suspicious</span>
                    ) : (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Normal</span>
                    )}
                  </td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Posts</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.posts}</td>
                  <td className="px-4 py-3"><span className="text-gray-600">{data.profile_data.posts > 20 ? '📝 Active' : '😴 Inactive'}</span></td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Engagement Rate</td>
                  <td className="px-4 py-3 text-right">{(data.profile_data.engagement_rate * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3">
                    {data.profile_data.engagement_rate > 0.05 ? (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Healthy</span>
                    ) : (
                      <span className="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded">⚠️ Low</span>
                    )}
                  </td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Account Age</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.account_age_days} days</td>
                  <td className="px-4 py-3">
                    {data.profile_data.account_age_days > 180 ? (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Established</span>
                    ) : (
                      <span className="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded">⚠️ New</span>
                    )}
                  </td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Profile Picture</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.has_profile_pic ? 'Yes' : 'No'}</td>
                  <td className="px-4 py-3">
                    {data.profile_data.has_profile_pic ? (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Present</span>
                    ) : (
                      <span className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded">✗ Missing</span>
                    )}
                  </td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-dark">Bio Length</td>
                  <td className="px-4 py-3 text-right">{data.profile_data.bio_length} characters</td>
                  <td className="px-4 py-3">
                    {data.profile_data.bio_length > 20 ? (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">✓ Detailed</span>
                    ) : (
                      <span className="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded">⚠️ Minimal</span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Detailed Insights */}
        <div className="mt-12 bg-white rounded-2xl p-8 border-2 border-gray-200 mb-8">
          <h3 className="text-2xl font-bold text-dark mb-6">📋 Detailed Findings</h3>
          <div className="space-y-6">
            {/* Bar Chart Insights */}
            <div className="pb-6 border-b">
              <h4 className="font-semibold text-dark mb-2">📊 Metrics Overview</h4>
              <p className="text-gray-700">{explanations.bar}</p>
            </div>

            {/* Radar Chart Insights */}
            <div className="pb-6 border-b">
              <h4 className="font-semibold text-dark mb-2">📈 Account Health</h4>
              <p className="text-gray-700">{explanations.radar}</p>
            </div>

            {/* Line Chart Insights */}
            <div className="pb-6 border-b">
              <h4 className="font-semibold text-dark mb-2">📉 Engagement Trend</h4>
              <p className="text-gray-700">{explanations.line}</p>
            </div>

            {/* Doughnut Chart Insights */}
            <div>
              <h4 className="font-semibold text-dark mb-2">🎯 Score Breakdown</h4>
              <p className="text-gray-700">{explanations.doughnut}</p>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className={`mt-12 p-8 ${data.prediction === 1 || data.prediction === 'FAKE' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'} border-2 border-l-4 rounded-xl`}>
          <h3 className="text-2xl font-bold text-dark mb-4">
            {data.prediction === 1 || data.prediction === 'FAKE' ? '🚨 Why This Account is Likely FAKE' : '✓ Why This Account is Likely REAL'}
          </h3>
          <p className="text-gray-700 leading-relaxed mb-4">
            {data.prediction === 1 || data.prediction === 'FAKE'
              ? `Based on the comprehensive analysis, this account shows ${Math.round(data.confidence * 100)}% likelihood of being fake. Multiple red flags were detected including: unusual engagement patterns, suspicious follower/following ratios, incomplete profile information, and behavior inconsistent with genuine accounts. These factors combined indicate artificial activity, bot-like behavior, or fraudulent account practices.`
              : `Based on the comprehensive analysis, this account shows ${Math.round(data.confidence * 100)}% likelihood of being authentic. The profile demonstrates healthy engagement rates, realistic growth patterns, complete profile information, authentic interaction patterns, and behavior consistent with genuine users. These indicators collectively suggest a legitimate Instagram account.`}
          </p>
          <div className="mt-6 p-4 bg-white rounded-lg">
            <p className="text-sm font-semibold text-gray-600 mb-2">CONFIDENCE SCORE:</p>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div 
                className={`h-full ${data.prediction === 1 || data.prediction === 'FAKE' ? 'bg-danger' : 'bg-success'}`}
                style={{ width: `${data.confidence * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-600 mt-2">{Math.round(data.confidence * 100)}% - {data.prediction === 1 || data.prediction === 'FAKE' ? 'High confidence this is a fake account' : 'High confidence this is a real account'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
