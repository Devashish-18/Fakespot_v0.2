import React from 'react';

export default function ChartCard({ title, children, explanation }) {
  return (
    <div className="chart-container slide-up">
      <h3 className="text-lg font-bold text-dark mb-4">{title}</h3>
      <div className="mb-6 overflow-x-auto">
        {children}
      </div>
      {explanation && (
        <div className="mt-4 p-4 bg-blue-50 border-l-4 border-primary rounded">
          <p className="text-sm text-gray-700">
            <strong className="text-primary">💡 Insight:</strong> {explanation}
          </p>
        </div>
      )}
    </div>
  );
}
