type Props = {
  error: string;
  onRetry?: () => void;
  onDismiss?: () => void;
};

export default function ErrorAlert({ error, onRetry, onDismiss }: Props) {
  return (
    <div className="card !border-rose-500/40 !bg-rose-500/10 flex items-start gap-3 p-3">
      <div className="flex-shrink-0 text-rose-400 text-lg leading-none mt-0.5">⚠</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-rose-200">{error}</p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-xs font-medium text-rose-300 hover:text-rose-200 px-2 py-1 rounded hover:bg-rose-500/20 transition-colors"
            aria-label="Retry"
          >
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-xs text-rose-400/60 hover:text-rose-400 leading-none w-5 h-5 flex items-center justify-center"
            aria-label="Dismiss error"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}
