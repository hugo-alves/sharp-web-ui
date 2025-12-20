import { useCallback, useEffect, useState } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { ProcessingQueue } from './components/ProcessingQueue';
import { GaussianViewer } from './components/GaussianViewer';
import { ResultsGallery } from './components/ResultsGallery';
import {
  type Job,
  getJobStatus,
  getDownloadAllUrl,
  clearAllJobs,
  listJobs,
} from './api/client';

function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load existing jobs on page load
  useEffect(() => {
    async function loadExistingJobs() {
      try {
        const existingJobs = await listJobs();
        setJobs(existingJobs);
      } catch (error) {
        console.error('Failed to load existing jobs:', error);
      } finally {
        setIsLoading(false);
      }
    }

    loadExistingJobs();
  }, []);

  const handleUploadComplete = useCallback((newJobs: Job[]) => {
    setJobs((prev) => [...newJobs, ...prev]);
  }, []);

  const handleSelectJob = useCallback((job: Job) => {
    setSelectedJob(job);
  }, []);

  const handleJobDeleted = useCallback((jobId: string) => {
    setJobs((prev) => prev.filter((j) => j.id !== jobId));
    if (selectedJob?.id === jobId) {
      setSelectedJob(null);
    }
  }, [selectedJob]);

  const handleClearAll = useCallback(async () => {
    if (confirm('Delete all results?')) {
      try {
        await clearAllJobs();
        setJobs([]);
        setSelectedJob(null);
      } catch (error) {
        console.error('Failed to clear jobs:', error);
      }
    }
  }, []);

  useEffect(() => {
    const pendingOrProcessing = jobs.filter(
      (j) => j.status === 'pending' || j.status === 'processing'
    );

    if (pendingOrProcessing.length === 0) return;

    const interval = setInterval(async () => {
      const updates = await Promise.all(
        pendingOrProcessing.map((job) => getJobStatus(job.id).catch(() => job))
      );

      setJobs((prev) =>
        prev.map((job) => {
          const update = updates.find((u) => u.id === job.id);
          return update || job;
        })
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [jobs]);

  const completedCount = jobs.filter((j) => j.status === 'completed').length;

  return (
    <div className="min-h-screen bg-gray-100 overflow-x-hidden">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <h1 className="text-xl sm:text-2xl font-bold text-gray-900">SHARP</h1>
              <p className="text-xs sm:text-sm text-gray-500 truncate">
                Single-image 3D Gaussian Splat Generation
              </p>
            </div>

            {completedCount > 0 && (
              <div className="flex items-center gap-2 sm:gap-3">
                <a
                  href={getDownloadAllUrl()}
                  className="inline-flex items-center justify-center gap-2 px-3 py-2.5 sm:px-4 sm:py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium touch-manipulation"
                  style={{ minHeight: '44px' }}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <span className="hidden sm:inline">Download All ({completedCount})</span>
                  <span className="sm:hidden">{completedCount}</span>
                </a>
                <button
                  onClick={handleClearAll}
                  className="inline-flex items-center justify-center gap-2 px-3 py-2.5 sm:px-4 sm:py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium touch-manipulation"
                  style={{ minHeight: '44px' }}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span className="hidden sm:inline">Clear All</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-2" />
              <p className="text-gray-500 text-sm">Loading...</p>
            </div>
          </div>
        ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <ImageUploader onUploadComplete={handleUploadComplete} />

            <ProcessingQueue
              jobs={jobs}
              onSelectJob={handleSelectJob}
              selectedJobId={selectedJob?.id}
            />

            <ResultsGallery
              jobs={jobs}
              onSelectJob={handleSelectJob}
              onJobDeleted={handleJobDeleted}
              selectedJobId={selectedJob?.id}
            />
          </div>

          <div className="lg:sticky lg:top-8 h-fit">
            {selectedJob && selectedJob.status === 'completed' ? (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700">
                      3D Preview
                    </h3>
                    <p className="text-xs text-gray-500">{selectedJob.filename}</p>
                  </div>
                  <button
                    onClick={() => setSelectedJob(null)}
                    className="p-1 text-gray-400 hover:text-gray-600 rounded"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <GaussianViewer
                  jobId={selectedJob.id}
                  className="h-[60vh] min-h-[300px] max-h-[500px] sm:h-[400px] lg:h-[500px]"
                />
                <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
                  <a
                    href={`/api/proxy/download/${selectedJob.id}`}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium w-full justify-center"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Download PLY File
                  </a>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 h-[60vh] min-h-[300px] max-h-[500px] sm:h-[400px] lg:h-[500px] flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                  </svg>
                  <p className="text-sm">Select a completed result to view in 3D</p>
                </div>
              </div>
            )}
          </div>
        </div>
        )}
      </main>

      <footer className="border-t border-gray-200 bg-white mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            SHARP - Monocular View Synthesis via 3D Gaussian Splatting |{' '}
            <a
              href="https://github.com/apple/ml-sharp"
              className="text-blue-600 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
