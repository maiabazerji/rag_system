type Props = { className?: string };

export const SparkleIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3z" strokeLinejoin="round"/>
    <path d="M19 14l.9 2.4 2.4.9-2.4.9L19 20.6 18.1 18.2l-2.4-.9 2.4-.9L19 14z" strokeLinejoin="round"/>
  </svg>
);

export const ChatIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M21 12a8 8 0 1 1-3.2-6.4L21 4l-1.6 3.2A7.96 7.96 0 0 1 21 12z" strokeLinejoin="round"/>
  </svg>
);

export const CompareIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <rect x="3" y="4" width="7" height="16" rx="2"/>
    <rect x="14" y="4" width="7" height="16" rx="2"/>
  </svg>
);

export const BeakerIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M9 3h6v4l5 11a2 2 0 0 1-1.8 2.9H5.8A2 2 0 0 1 4 18l5-11V3z" strokeLinejoin="round"/>
    <path d="M7 14h10"/>
  </svg>
);

export const TrendIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M3 17l6-6 4 4 8-8" strokeLinejoin="round" strokeLinecap="round"/>
    <path d="M14 7h7v7" strokeLinejoin="round" strokeLinecap="round"/>
  </svg>
);

export const UploadIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M12 16V4M6 10l6-6 6 6" strokeLinejoin="round" strokeLinecap="round"/>
    <path d="M4 20h16" strokeLinecap="round"/>
  </svg>
);

export const SendIcon = ({ className = "w-4 h-4" }: Props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M4 12l16-8-6 18-3-7-7-3z" strokeLinejoin="round" strokeLinecap="round"/>
  </svg>
);
