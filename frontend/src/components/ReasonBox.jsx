import React from 'react';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

export default function ReasonBox({ reason }) {
  const getSeverityColor = (impact) => {
    switch (impact.toLowerCase()) {
      case 'high':
        return 'border-danger bg-red-50';
      case 'medium':
        return 'border-warning bg-yellow-50';
      case 'low':
        return 'border-success bg-green-50';
      default:
        return 'border-primary bg-indigo-50';
    }
  };

  const getSeverityIcon = (impact) => {
    switch (impact.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="text-danger w-5 h-5" />;
      case 'medium':
        return <Info className="text-warning w-5 h-5" />;
      case 'low':
        return <CheckCircle className="text-success w-5 h-5" />;
      default:
        return <Info className="text-primary w-5 h-5" />;
    }
  };

  return (
    <div className={`${getSeverityColor(reason.impact)} border-l-4 rounded-lg p-4 fade-in`}>
      <div className="flex gap-3">
        {getSeverityIcon(reason.impact)}
        <div className="flex-1">
          <h4 className="font-semibold text-dark mb-1">{reason.signal}</h4>
          <p className="text-sm text-gray-700">{reason.detail}</p>
          <div className="mt-2 inline-block px-2 py-1 bg-white rounded text-xs font-medium">
            Impact: <span className="capitalize">{reason.impact}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
