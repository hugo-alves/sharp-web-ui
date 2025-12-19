import { type Job, getThumbnailUrl } from '../api/client';

interface ProcessingQueueProps {
  jobs: Job[];
  onSelectJob: (job: Job) => void;
  selectedJobId?: string;
}

function StatusBadge({ status }: { status: Job['status'] }) {
  const styles = {
    pending: 'bg-gray-100 text-gray-700',
    processing: 'bg-blue-100 text-blue-700',
    completed: 'bg-green-100 text-green-700',
    failed: 'bg-red-100 text-red-700',
  };

  const labels = {
    pending: 'Pending',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
  };

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status]}`}>
      {labels[status]}
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-blue-500"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

export function ProcessingQueue({ jobs, onSelectJob, selectedJobId }: ProcessingQueueProps) {
  if (jobs.length === 0) {
    return null;
  }

  const pendingOrProcessing = jobs.filter(
    (j) => j.status === 'pending' || j.status === 'processing'
  );

  if (pendingOrProcessing.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <Spinner />
          Processing Queue ({pendingOrProcessing.length})
        </h3>
      </div>

      <ul className="divide-y divide-gray-100">
        {pendingOrProcessing.map((job) => (
          <li
            key={job.id}
            onClick={() => onSelectJob(job)}
            className={`
              flex items-center gap-4 p-4 cursor-pointer transition-colors
              ${selectedJobId === job.id ? 'bg-blue-50' : 'hover:bg-gray-50'}
            `}
          >
            <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
              <img
                src={getThumbnailUrl(job.id)}
                alt={job.filename}
                className="w-full h-full object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>

            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {job.filename}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(job.created_at).toLocaleTimeString()}
              </p>
            </div>

            <StatusBadge status={job.status} />
          </li>
        ))}
      </ul>
    </div>
  );
}
