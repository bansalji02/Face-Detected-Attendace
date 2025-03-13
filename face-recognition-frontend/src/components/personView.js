import React from 'react';

const PersonView = ({ selectedPerson, personRecords, formatTimestamp }) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">Records for {selectedPerson}</h2>
      
      {personRecords.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border-collapse">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Camera</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Timestamp</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Location</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {personRecords.map((record, index) => (
                <tr key={index} className={`hover:bg-gray-50 transition-colors duration-150
                  ${record.camera === 'Cam-01' ? 'bg-green-50' : 'bg-red-50'}`}>
                  <td className="py-3 px-4 border-b">
                    <div className="flex items-center">
                      <span className={`w-3 h-3 rounded-full mr-2 ${
                        record.camera === 'Cam-01' ? 'bg-green-500' : 'bg-red-500'
                      }`}></span>
                      {record.camera === 'Cam-01' ? 'Entry (Cam-01)' : 'Exit (Cam-02)'}
                    </div>
                  </td>
                  <td className="py-3 px-4 border-b">{formatTimestamp(record.timestamp)}</td>
                  <td className="py-3 px-4 border-b">{record.location}</td>
                  <td className="py-3 px-4 border-b">
                    <div className="flex items-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className={`h-2.5 rounded-full ${record.confidence > 0.7 ? 'bg-green-600' : 'bg-yellow-500'}`} 
                          style={{ width: `${record.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="ml-2">{(record.confidence * 100).toFixed(2)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="p-8 text-center bg-gray-50 rounded-md">
          <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 12H4" />
          </svg>
          <p className="text-gray-500">No records found for this person within the selected date range.</p>
        </div>
      )}
    </div>
  );
};

export default PersonView;