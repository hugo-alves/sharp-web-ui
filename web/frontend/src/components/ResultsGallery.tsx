import { type Job, getThumbnailUrl, getDownloadUrl, deleteJob } from '../api/client';

interface ResultsGalleryProps {
  jobs: Job[];
  onSelectJob: (job: Job) => void;
  onJobDeleted: (jobId: string) => void;
  selectedJobId?: string;
}

function DownloadIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}

function ViewIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
      />
    </svg>
  );
}

export function ResultsGallery({
  jobs,
  onSelectJob,
  onJobDeleted,
  selectedJobId,
}: ResultsGalleryProps) {
  const completedJobs = jobs.filter((j) => j.status === 'completed');
  const failedJobs = jobs.filter((j) => j.status === 'failed');

  const handleDelete = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (confirm('Delete this result?')) {
      try {
        await deleteJob(jobId);
        onJobDeleted(jobId);
      } catch (error) {
        console.error('Failed to delete job:', error);
      }
    }
  };

  const handleDownload = (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    window.open(getDownloadUrl(jobId), '_blank');
  };

  if (completedJobs.length === 0 && failedJobs.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6">
      {completedJobs.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Results ({completedJobs.length})
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {completedJobs.map((job) => (
              <div
                key={job.id}
                onClick={() => onSelectJob(job)}
                className={`
                  group relative bg-white rounded-xl overflow-hidden shadow-sm
                  border-2 transition-all cursor-pointer
                  ${selectedJobId === job.id
                    ? 'border-blue-500 ring-2 ring-blue-200'
                    : 'border-transparent hover:border-gray-200 hover:shadow-md'
                  }
                `}
              >
                <div className="aspect-square bg-gray-100">
                  <img
                    src={getThumbnailUrl(job.id)}
                    alt={job.filename}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="p-3">
                  <p className="text-sm font-medium text-gray-800 truncate">
                    {job.filename}
                  </p>
                  {job.result && (
                    <p className="text-xs text-gray-500 mt-1">
                      {job.result.num_gaussians.toLocaleString()} gaussians
                    </p>
                  )}
                </div>

                <div className="absolute top-2 right-2 flex gap-1 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => onSelectJob(job)}
                    className="p-3 sm:p-2 bg-white/90 hover:bg-white rounded-lg shadow-sm text-gray-600 hover:text-blue-600 touch-manipulation"
                    title="View 3D"
                  >
                    <ViewIcon />
                  </button>
                  <button
                    onClick={(e) => handleDownload(e, job.id)}
                    className="p-3 sm:p-2 bg-white/90 hover:bg-white rounded-lg shadow-sm text-gray-600 hover:text-green-600 touch-manipulation"
                    title="Download PLY"
                  >
                    <DownloadIcon />
                  </button>
                  <button
                    onClick={(e) => handleDelete(e, job.id)}
                    className="p-3 sm:p-2 bg-white/90 hover:bg-white rounded-lg shadow-sm text-gray-600 hover:text-red-600 touch-manipulation"
                    title="Delete"
                  >
                    <TrashIcon />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {failedJobs.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-red-700 mb-4">
            Failed ({failedJobs.length})
          </h3>

          <div className="space-y-2">
            {failedJobs.map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between bg-red-50 border border-red-200 rounded-lg p-3"
              >
                <div>
                  <p className="text-sm font-medium text-red-800">{job.filename}</p>
                  <p className="text-xs text-red-600 mt-1">{job.error}</p>
                </div>
                <button
                  onClick={(e) => handleDelete(e, job.id)}
                  className="p-2 text-red-600 hover:text-red-800 hover:bg-red-100 rounded-lg"
                >
                  <TrashIcon />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
