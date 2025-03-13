import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CameraView from '../components/cameraView';
import PersonView from '../components/personView';
import ReportView from '../components/reportView';
import { formatTimestamp, formatDuration } from '../utils/format';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

function PresenceDashboard() {
  const [cameraRecords, setCameraRecords] = useState([]);
  const [personRecords, setPersonRecords] = useState([]);
  const [presenceDurations, setPresenceDurations] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('Cam-01');
  const [selectedPerson, setSelectedPerson] = useState('');
  const [searchPerson, setSearchPerson] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fromDate, setFromDate] = useState(new Date().toISOString().slice(0, 10));
  const [toDate, setToDate] = useState(new Date().toISOString().slice(0, 10));
  const [appliedFromDate, setAppliedFromDate] = useState(new Date().toISOString().slice(0, 10));
  const [appliedToDate, setAppliedToDate] = useState(new Date().toISOString().slice(0, 10));
  const [showDurationReport, setShowDurationReport] = useState(false);
  const [activeView, setActiveView] = useState('cameras'); // 'cameras', 'person', 'report'
  const [singleDayReport, setSingleDayReport] = useState(false);
  const [reportDate, setReportDate] = useState(new Date().toISOString().slice(0, 10));

  // Fetch camera records when camera or applied date range changes
  useEffect(() => {
    if (selectedCamera && activeView === 'cameras') {
      const fetchCameraRecords = async () => {
        try {
          setLoading(true);
          const response = await axios.get(`${API_BASE_URL}/camera/${selectedCamera}?fromDate=${appliedFromDate}&toDate=${appliedToDate}`);
          if (response.data.status === 'success') {
            setCameraRecords(response.data.records);
          } else {
            setError(`Failed to fetch camera records: ${response.data.message}`);
          }
          setLoading(false);
        } catch (err) {
          setError(`Failed to fetch camera records: ${err.message}`);
          setLoading(false);
        }
      };

      fetchCameraRecords();
    }
  }, [selectedCamera, appliedFromDate, appliedToDate, activeView]);

  // Fetch person records when a person is selected and date range is applied
  useEffect(() => {
    if (selectedPerson && activeView === 'person') {
      const fetchPersonRecords = async () => {
        try {
          setLoading(true);
          const response = await axios.get(`${API_BASE_URL}/person/${selectedPerson}?fromDate=${appliedFromDate}&toDate=${appliedToDate}`);
          if (response.data.status === 'success') {
            setPersonRecords(response.data.records);
            calculatePresenceDuration(response.data.records);
          } else {
            setError(`Failed to fetch person records: ${response.data.message}`);
          }
          setLoading(false);
        } catch (err) {
          setError(`Failed to fetch person records: ${err.message}`);
          setLoading(false);
        }
      };

      fetchPersonRecords();
    }
  }, [selectedPerson, appliedFromDate, appliedToDate, activeView]);

  // Fetch all persons' presence durations for date range
  const fetchAllPresenceDurations = async () => {
    try {
      setLoading(true);
      setShowDurationReport(true);
      setActiveView('report');
  
      const startDate = singleDayReport ? reportDate : appliedFromDate;
      const endDate = singleDayReport ? reportDate : appliedToDate;
  
      // First get all people who entered during the time period
      const entryResponse = await axios.get(`${API_BASE_URL}/camera/Cam-01?fromDate=${startDate}&toDate=${endDate}`);
  
      if (entryResponse.data.status === 'success') {
        const entryCameraRecords = entryResponse.data.records;
        const uniquePeople = [...new Set(entryCameraRecords.map((record) => record.person))];
  
        // For each person, get their records and calculate durations
        const durationsPromises = uniquePeople.map(async (person) => {
          const personResponse = await axios.get(`${API_BASE_URL}/person/${person}?fromDate=${startDate}&toDate=${endDate}`);
          if (personResponse.data.status === 'success') {
            return calculatePresenceDurationForPerson(personResponse.data.records, person);
          }
          return [];
        });
  
        const allDurations = await Promise.all(durationsPromises);
        // Flatten the array of arrays
        setPresenceDurations(allDurations.flat());
      }
  
      setLoading(false);
    } catch (err) {
      setError(`Failed to fetch presence durations: ${err.message}`);
      setLoading(false);
    }
  };

  // Calculate presence durations for a single person
  const calculatePresenceDuration = (records) => {
    const durations = calculatePresenceDurationForPerson(records, selectedPerson);
    setPresenceDurations(durations);
  };

  // Helper function to calculate presence durations for a person
  const calculatePresenceDurationForPerson = (records, personName) => {
    // Sort records by timestamp
    const sortedRecords = [...records].sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );
    
    const durations = [];
    let entryTime = null;
    
    // Process records to match entry/exit pairs
    for (let i = 0; i < sortedRecords.length; i++) {
      const record = sortedRecords[i];
      
      if (record.camera === 'Cam-01' && !entryTime) {
        // Entry detected
        entryTime = new Date(record.timestamp);
      } else if (record.camera === 'Cam-02' && entryTime) {
        // Exit detected - calculate duration and reset entry time
        const exitTime = new Date(record.timestamp);
        const durationMs = exitTime - entryTime;
        
        durations.push({
          person: personName,
          entryTime,
          exitTime,
          durationMs,
          entryTimestamp: record.timestamp,
          exitTimestamp: sortedRecords[i].timestamp
        });
        
        entryTime = null;
      }
    }
    
    // Handle case where person entered but hasn't exited yet
    if (entryTime) {
      durations.push({
        person: personName,
        entryTime,
        exitTime: null,
        durationMs: null,
        entryTimestamp: entryTime.toISOString(),
        exitTimestamp: null
      });
    }
    
    return durations;
  };

  // Handle camera selection
  const handleCameraChange = (cam) => {
    setSelectedCamera(cam);
    setSelectedPerson('');
    setShowDurationReport(false);
    setActiveView('cameras');
  };

  // Handle person search
  const handlePersonSearch = (e) => {
    e.preventDefault();
    if (searchPerson.trim()) {
      setSelectedPerson(searchPerson);
      setActiveView('person');
      setShowDurationReport(false);
    }
  };

  // Apply date range when OK button is clicked
  const applyDateRange = () => {
    setAppliedFromDate(fromDate);
    setAppliedToDate(toDate);
    setShowDurationReport(false);
  };

  // Generate a specific date report
  const generateDailyReport = () => {
    setSingleDayReport(true);
    fetchAllPresenceDurations();
  };

  // Generate a date range report
  const generateRangeReport = () => {
    setSingleDayReport(false);
    fetchAllPresenceDurations();
  };

  // Function to switch to report view
  const switchToReportView = () => {
    setActiveView('report');
    setSelectedCamera('');
    setSelectedPerson('');
  };

  // Reset view to main dashboard
  const resetView = () => {
    setActiveView('cameras');
    setSelectedCamera('Cam-01');
    setSelectedPerson('');
    setShowDurationReport(false);
  };



    // Function to handle CSV download
    const handleDownloadCSV = () => {
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
  
      // Convert personStats to CSV
      const csvRows = [];
      csvRows.push(['Person', 'Total Visits', 'Total Time in Office', 'Average Duration', 'Status']);
  
      Object.keys(personStats).forEach((person) => {
        const stats = personStats[person];
        csvRows.push([
          person,
          stats.visits,
          formatDuration(stats.totalDuration),
          stats.completedVisits > 0
            ? formatDuration(stats.totalDuration / stats.completedVisits)
            : 'N/A',
          stats.inProgress ? 'Currently in office' : 'Not present',
        ]);
      });
  
      const csvContent = csvRows.map((row) => row.join(',')).join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'person_wise_summary.csv';
      link.click();
      URL.revokeObjectURL(url);
    };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4">
      <div className="max-w-6xl mx-auto">
        <header className="bg-white shadow-md rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-blue-800 mb-2">Office Presence Dashboard</h1>
          <p className="text-gray-600">Monitor entry/exit and calculate time spent in office</p>
          
          {/* Main Navigation */}
          <div className="mt-6 flex flex-wrap gap-2">
            <button
              onClick={resetView}
              className={`px-4 py-2 rounded-md transition-all duration-200 ${
                activeView === 'cameras' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              Camera View
            </button>
            
            <button
              onClick={switchToReportView}
              className={`px-4 py-2 rounded-md transition-all duration-200 ${
                activeView === 'report' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              Reports
            </button>
          </div>
        </header>
        
        {/* Date range selector - visible only in cameras and person view */}
        {(activeView === 'cameras' || activeView === 'person') && (
          <div className="bg-white shadow-md rounded-lg p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Date Range Selection</h2>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center">
                <span className="font-medium mr-2">From</span>
                <input
                  type="date"
                  value={fromDate}
                  onChange={(e) => setFromDate(e.target.value)}
                  max={new Date().toISOString().split('T')[0]}
                  className="p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div className="flex items-center">
                <span className="font-medium mr-2">To</span>
                <input
                  type="date"
                  value={toDate}
                  onChange={(e) => setToDate(e.target.value)}
                  max={new Date().toISOString().split('T')[0]}
                  className="p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <button
                type="button"
                onClick={applyDateRange}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors duration-200 shadow-sm"
              >
                Apply
              </button>
            </div>
            
            <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-200">
              <p className="text-blue-800 text-sm">
                Currently viewing data from <strong>{new Date(appliedFromDate).toLocaleDateString()}</strong> to <strong>{new Date(appliedToDate).toLocaleDateString()}</strong>
              </p>
            </div>
          </div>
        )}
        
        {/* Camera and Person Search views */}
        {(activeView === 'cameras' || activeView === 'person') && (
          <div className="bg-white shadow-md rounded-lg p-6 mb-6">
            <div className="flex flex-col md:flex-row gap-6">
              {/* Camera View Controls */}
              <div className="flex-1">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">View by Camera</h2>
                <div className="flex space-x-2">
                  <button 
                    onClick={() => handleCameraChange('Cam-01')} 
                    className={`px-4 py-2 rounded-md transition-colors duration-200 ${
                      selectedCamera === 'Cam-01' && activeView === 'cameras'
                        ? 'bg-green-600 text-white shadow-sm' 
                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                    }`}
                  >
                    Entry Camera (Cam-01)
                  </button>
                  <button 
                    onClick={() => handleCameraChange('Cam-02')} 
                    className={`px-4 py-2 rounded-md transition-colors duration-200 ${
                      selectedCamera === 'Cam-02' && activeView === 'cameras' 
                        ? 'bg-red-600 text-white shadow-sm' 
                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                    }`}
                  >
                    Exit Camera (Cam-02)
                  </button>
                </div>
              </div>
              
              {/* Person Search Controls */}
              <div className="flex-1">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Search by Person</h2>
                <form onSubmit={handlePersonSearch} className="flex">
                  <input
                    type="text"
                    value={searchPerson}
                    onChange={(e) => setSearchPerson(e.target.value)}
                    placeholder="Enter person name"
                    className="flex-1 p-2 border rounded-l-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <button 
                    type="submit" 
                    className="bg-blue-600 text-white px-4 py-2 rounded-r-md hover:bg-blue-700 transition-colors duration-200 shadow-sm"
                  >
                    Search
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}
{/* // Report Generation Controls*/}
{activeView === 'report' && !loading && (
  <div className="bg-white shadow-md rounded-lg p-6 mb-6">
    <h2 className="text-xl font-semibold text-gray-800 mb-4">Office Time Reports</h2>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
      {/* Single Day Report */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <h3 className="text-lg font-semibold mb-3 text-blue-800">Single Day Report</h3>
        <div className="flex flex-col space-y-3">
          <div className="flex items-center">
            <span className="font-medium mr-2">Select Date:</span>
            <input
              type="date"
              value={reportDate}
              onChange={(e) => setReportDate(e.target.value)}
              max={new Date().toISOString().split('T')[0]}
              className="p-2 border rounded-md flex-1 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <button
            onClick={generateDailyReport}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors duration-200 shadow-sm"
          >
            Generate Daily Report
          </button>
        </div>
      </div>

      {/* Date Range Report */}
      <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
        <h3 className="text-lg font-semibold mb-3 text-indigo-800">Date Range Report</h3>
        <div className="flex flex-col space-y-3">
          <div className="flex items-center">
            <span className="font-medium mr-2">From:</span>
            <input
              type="date"
              value={appliedFromDate}
              onChange={(e) => setAppliedFromDate(e.target.value)}
              max={new Date().toISOString().split('T')[0]}
              className="p-2 border rounded-md flex-1 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div className="flex items-center">
            <span className="font-medium mr-2">To:</span>
            <input
              type="date"
              value={appliedToDate}
              onChange={(e) => setAppliedToDate(e.target.value)}
              max={new Date().toISOString().split('T')[0]}
              className="p-2 border rounded-md flex-1 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <button
            onClick={generateRangeReport}
            className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors duration-200 shadow-sm"
          >
            Generate Range Report
          </button>
        </div>
      </div>
    </div>
  </div>
)}
{/* 
         Display presence durations report
         {showDurationReport && presenceDurations.length > 0 && activeView === 'report' && (
          <ReportView
            presenceDurations={presenceDurations}
            singleDayReport={singleDayReport}
            reportDate={reportDate}
            appliedFromDate={appliedFromDate}
            appliedToDate={appliedToDate}
            formatTimestamp={formatTimestamp}
            formatDuration={formatDuration}
            onDownloadCSV={handleDownloadCSV}
          />
        )} */}


        {/* Error message */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-md mb-6 shadow-sm">
            <div className="flex">
              <div className="py-1 mr-2">
                <svg className="h-6 w-6 text-red-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <p className="font-bold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="bg-white shadow-md rounded-lg p-8 mb-6 flex justify-center items-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading data...</p>
            </div>
          </div>
        )}

        {/* Display camera records */}
        {selectedCamera && activeView === 'cameras' && !loading && (
          <CameraView
            selectedCamera={selectedCamera}
            cameraRecords={cameraRecords}
            formatTimestamp={formatTimestamp}
          />
        )}

        {/* Display person records */}
        {selectedPerson && activeView === 'person' && !loading && (
          <PersonView
            selectedPerson={selectedPerson}
            personRecords={personRecords}
            formatTimestamp={formatTimestamp}
          />
        )}

        {/* Display presence durations report */}
        {showDurationReport && presenceDurations.length > 0 && activeView === 'report' && (
          <ReportView
            presenceDurations={presenceDurations}
            singleDayReport={singleDayReport}
            reportDate={reportDate}
            appliedFromDate={appliedFromDate}
            appliedToDate={appliedToDate}
            formatTimestamp={formatTimestamp}
            formatDuration={formatDuration}
          />
        )}

        {/* Footer */}
        <footer className="py-4 text-center text-gray-500 text-sm">
          <p>Office Presence Monitoring System &copy; {new Date().getFullYear()}</p>
        </footer>
      </div>
    </div>
  );
}

export default PresenceDashboard;