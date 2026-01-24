import React from 'react';
import { formatCountSafe } from '../utils/countFormatter';

export default function MetricCard({ label, value, unit = '', icon: Icon, color = 'primary', formatAsCount = false }) {
  const colorClasses = {
    primary: 'border-primary bg-indigo-50',
    secondary: 'border-secondary bg-pink-50',
    success: 'border-success bg-green-50',
    warning: 'border-warning bg-yellow-50',
    danger: 'border-danger bg-red-50',
  };

  const iconColorClasses = {
    primary: 'text-primary',
    secondary: 'text-secondary',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
  };

  // Format the value if it's a count (number > 1000)
  const displayValue = formatAsCount && typeof value === 'number' 
    ? formatCountSafe(value, value)
    : value;

  return (
    <div className={`${colorClasses[color]} border-l-4 rounded-lg p-6 fade-in`}>
      <div className="flex items-start justify-between mb-2">
        <h3 className="text-gray-600 font-medium">{label}</h3>
        {Icon && <Icon className={`${iconColorClasses[color]} w-5 h-5`} />}
      </div>
      <div className="text-3xl font-bold text-dark">
        {displayValue}
        <span className="text-sm text-gray-500 ml-1">{unit}</span>
      </div>
    </div>
  );
}
