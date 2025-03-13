import React from 'react';
import { formatTimestamp, formatDuration } from '../utils/format';

const ReportView = ({
  presenceDurations,
  singleDayReport,
  reportDate,
  appliedFromDate,
  appliedToDate,
  formatTimestamp,
  formatDuration,
  onDownloadCSV,
}) => {
  // Calculate per-person stats
  const personStats = {};
  presenceDurations.forEach((duration) => {
    if (!personStats[duration.person]) {
      personStats[duration.person] = {
        visits: 0,
        totalDuration: 0,
        completedVisits: 0,
        inProgress: false,
      };
    }

    personStats[duration.person].visits++;

    if (duration.durationMs) {
      personStats[duration.person].totalDuration += duration.durationMs;
      personStats[duration.person].completedVisits++;
    } else {
      personStats[duration.person].inProgress = true;
    }
  });

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">
        {singleDayReport
          ? `Office Presence Report for ${new Date(reportDate).toLocaleDateString()}`
          : `Office Presence Report from ${new Date(appliedFromDate).toLocaleDateString()} to ${new Date(appliedToDate).toLocaleDateString()}`
        }
      </h2>

      {/* Summary statistics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg shadow-sm border border-blue-200">
          <p className="text-sm text-blue-700 mb-1">Total People</p>
          <p className="text-2xl font-bold text-blue-900">
            {[...new Set(presenceDurations.map((d) => d.person))].length}
          </p>
        </div>
        <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg shadow-sm border border-green-200">
          <p className="text-sm text-green-700 mb-1">Total Visits</p>
          <p className="text-2xl font-bold text-green-900">{presenceDurations.length}</p>
        </div>
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg shadow-sm border border-purple-200">
          <p className="text-sm text-purple-700 mb-1">Average Visit Duration</p>
          <p className="text-2xl font-bold text-purple-900">
            {formatDuration(
              presenceDurations
                .filter((d) => d.durationMs)
                .reduce((sum, d) => sum + d.durationMs, 0) /
              (presenceDurations.filter((d) => d.durationMs).length || 1)
            )}
          </p>
        </div>
        <div className="bg-gradient-to-r from-amber-50 to-amber-100 p-4 rounded-lg shadow-sm border border-amber-200">
          <p className="text-sm text-amber-700 mb-1">People Currently In</p>
          <p className="text-2xl font-bold text-amber-900">
            {presenceDurations.filter((d) => !d.exitTime).length}
          </p>
        </div>
      </div>

      {/* Person-wise aggregated data */}
      <div className="mt-8">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Person-wise Summary</h3>
          <button
            onClick={onDownloadCSV}
            className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors duration-200 shadow-sm"
          >
            Download CSV
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border-collapse">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Person</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Total Visits</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Total Time in Office</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Average Duration</th>
                <th className="py-3 px-4 border-b text-left font-semibold text-gray-700">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(personStats).map((person, index) => {
                const stats = personStats[person];
                return (
                  <tr key={index} className="hover:bg-gray-50 transition-colors duration-150">
                    <td className="py-3 px-4 border-b font-medium">{person}</td>
                    <td className="py-3 px-4 border-b">{stats.visits}</td>
                    <td className="py-3 px-4 border-b">{formatDuration(stats.totalDuration)}</td>
                    <td className="py-3 px-4 border-b">
                      {stats.completedVisits > 0
                        ? formatDuration(stats.totalDuration / stats.completedVisits)
                        : 'N/A'}
                    </td>
                    <td className="py-3 px-4 border-b">
                      {stats.inProgress ? (
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                          Currently in office
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs font-medium">
                          Not present
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ReportView;