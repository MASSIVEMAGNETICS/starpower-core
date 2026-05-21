
import React, { useEffect, useRef } from 'react';
import { TranscriptItem } from '../types';

interface LiveViewProps {
  transcripts: TranscriptItem[];
  status: string;
}

const LiveView: React.FC<LiveViewProps> = ({ transcripts, status }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when transcripts change
  useEffect(() => {
    if (transcripts.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcripts]);

  return (
    <div className="relative flex-1 w-full h-full overflow-hidden">
        {/* Top Fade Gradient */}
        <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-black to-transparent z-10 pointer-events-none" />

        {/* Bottom Fade Gradient - Taller to cover the control bar area */}
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-black via-black/80 to-transparent z-10 pointer-events-none" />

        {/* Scrollable Content Container */}
        <div className="w-full h-full overflow-y-auto custom-scrollbar">
            <div className="max-w-4xl mx-auto px-6 flex flex-col space-y-6 pb-48 pt-24">
                {transcripts.map((item, index) => (
                    <div 
                        key={item.id + index} 
                        className={`flex flex-col ${item.sender === 'user' ? 'items-end' : 'items-start'}`}
                    >
                        <div 
                            className={`text-3xl font-medium leading-relaxed max-w-[90%]
                            ${item.sender === 'user' 
                                ? 'text-zinc-500 text-right' 
                                : 'text-white text-left'
                            }`}
                        >
                            {item.text}
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    </div>
  );
};

export default LiveView;
