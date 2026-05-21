
import React, { useState, useRef, useEffect } from 'react';
import { AudioLines, Square, Loader2, ArrowUp, Paperclip, Image as ImageIcon, Globe } from 'lucide-react';
import { LiveStatus } from '../types';

interface ControlBarProps {
  status: LiveStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  onOpenSettings: () => void;
  onSendText: (text: string) => void;
  error: string | null;
}

const ControlBar: React.FC<ControlBarProps> = ({ status, onConnect, onDisconnect, onOpenSettings, onSendText, error }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState(0);
  const [mode, setMode] = useState<'voice' | 'text'>('voice');
  const [promptText, setPromptText] = useState('');
  
  // Startup Hint State
  const [showHint, setShowHint] = useState(false);

  const startYRef = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Animation Constants
  const THRESHOLD = typeof window !== 'undefined' ? window.innerHeight * 0.3 : 200; 

  // Automate the hint animation
  useEffect(() => {
    const startTimer = setTimeout(() => {
      setShowHint(true);
      
      // Fade back after holding
      const endTimer = setTimeout(() => {
        setShowHint(false);
      }, 2000);

      return () => clearTimeout(endTimer);
    }, 800);

    return () => clearTimeout(startTimer);
  }, []);
  
  // Calculate Opacities based on drag, mode, or hint
  const getOpacities = () => {
    // 1. Handle Automated Hint
    if (showHint) {
      return { icons: 0, prompt: 1 };
    }

    // 2. Handle Text Mode (Open) - allow drag interaction
    if (mode === 'text') {
        // If dragging down, animate opacity
        if (isDragging && dragOffset > -THRESHOLD) {
             const progress = Math.min(1, Math.max(0, (THRESHOLD + dragOffset) / THRESHOLD));
             // Inverse of the logic below:
             // 1.0 -> 0.5: Prompt fades out
             // 0.5 -> 0.0: Icons fade in
             let iconOpacity = 0;
             let promptOpacity = 0;

             if (progress >= 0.5) {
                 iconOpacity = 0;
                 promptOpacity = (progress - 0.5) * 2;
             } else {
                 iconOpacity = 1 - (progress * 2);
                 promptOpacity = 0;
             }
             return { icons: iconOpacity, prompt: promptOpacity };
        }
        return { icons: 0, prompt: 1 };
    }

    // 3. Handle Voice Mode (Closed) - allow drag interaction
    if (mode === 'voice') {
         if (isDragging && dragOffset < 0) {
            const progress = Math.min(1, Math.max(0, -dragOffset / THRESHOLD));
            let iconOpacity = 0;
            let promptOpacity = 0;

            // 0% to 50%: Icon fades out
            if (progress <= 0.5) {
                iconOpacity = 1 - (progress * 2); 
                promptOpacity = 0;
            } 
            // 50% to 100%: Prompt fades in
            else {
                iconOpacity = 0;
                promptOpacity = (progress - 0.5) * 2;
            }
            return { icons: iconOpacity, prompt: promptOpacity };
         }
         return { icons: 1, prompt: 0 };
    }

    return { icons: 1, prompt: 0 };
  };

  const { icons: iconOpacity, prompt: promptOpacity } = getOpacities();

  // Gesture Handlers for Sliding UP (On Icons)
  const handleIconTouchStart = (e: React.TouchEvent | React.MouseEvent) => {
    if (mode === 'text') return;
    setIsDragging(true);
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    startYRef.current = clientY;
  };

  const handleIconTouchMove = (e: React.TouchEvent | React.MouseEvent) => {
    if (!isDragging) return;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const deltaY = clientY - startYRef.current;
    
    if (deltaY < 0) {
      setDragOffset(deltaY);
    }
  };

  const handleIconTouchEnd = () => {
    setIsDragging(false);
    if (-dragOffset > THRESHOLD * 0.5) {
      setMode('text');
      setDragOffset(-THRESHOLD);
    } else {
      setMode('voice');
      setDragOffset(0);
    }
  };

  // Gesture Handlers for Sliding DOWN (On Prompt Bar)
  const handlePromptTouchStart = (e: React.TouchEvent | React.MouseEvent) => {
    if (mode !== 'text') return;
    setIsDragging(true);
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    startYRef.current = clientY;
  };

  const handlePromptTouchMove = (e: React.TouchEvent | React.MouseEvent) => {
    if (!isDragging) return;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const deltaY = clientY - startYRef.current;
    
    if (deltaY > 0) {
       const newOffset = -THRESHOLD + deltaY;
       if (newOffset <= 0) setDragOffset(newOffset);
    }
  };

  const handlePromptTouchEnd = () => {
    setIsDragging(false);
    if (-dragOffset < THRESHOLD * 0.7) {
      setMode('voice');
      setDragOffset(0);
    } else {
      setDragOffset(-THRESHOLD);
    }
  };

  const handleSend = () => {
    if (promptText.trim()) {
      onSendText(promptText);
      setPromptText('');
      setMode('voice');
      setDragOffset(0);
    }
  };

  // Determine active state for sequential transitions
  const showPrompt = showHint || mode === 'text' || (isDragging && -dragOffset > THRESHOLD * 0.5);

  return (
    <>
      {/* Prompt Bar */}
      <div 
        className="absolute bottom-6 left-4 right-4 bg-zinc-900 border border-zinc-800 shadow-2xl z-30 flex flex-col rounded-3xl overflow-hidden"
        style={{
            height: '170px', 
            opacity: promptOpacity,
            pointerEvents: (mode === 'text' || showHint) ? 'auto' : 'none',
            // Sequential Transition: Delay fade-in so it happens after icon fades out
            // Duration: 0.3s
            transition: isDragging ? 'none' : `opacity 0.3s ease-out ${showPrompt ? '0.3s' : '0s'}`
        }}
        onTouchStart={handlePromptTouchStart}
        onTouchMove={handlePromptTouchMove}
        onTouchEnd={handlePromptTouchEnd}
        onMouseDown={handlePromptTouchStart}
        onMouseMove={handlePromptTouchMove}
        onMouseUp={handlePromptTouchEnd}
        onMouseLeave={handlePromptTouchEnd}
      >
         {/* Drag Handle */}
         <div className="w-full h-6 flex items-center justify-center shrink-0 cursor-grab active:cursor-grabbing hover:bg-white/5">
             <div className="w-12 h-1 bg-zinc-700 rounded-full" />
         </div>

         <div className="flex-1 flex flex-col px-6 pb-6 pt-2">
            <textarea
                autoFocus={mode === 'text'}
                className="w-full flex-1 bg-transparent text-lg text-white placeholder:text-zinc-600 resize-none focus:outline-none font-medium mb-4"
                placeholder="Type a message..."
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                onKeyDown={(e) => {
                    if(e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                    }
                }}
            />

            <div className="h-12 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-2 text-zinc-400">
                    <button className="p-2 hover:bg-zinc-800 rounded-full transition-colors">
                        <Paperclip className="w-5 h-5" />
                    </button>
                    <button className="p-2 hover:bg-zinc-800 rounded-full transition-colors">
                        <ImageIcon className="w-5 h-5" />
                    </button>
                    <button className="p-2 hover:bg-zinc-800 rounded-full transition-colors">
                        <Globe className="w-5 h-5" />
                    </button>
                </div>

                <button 
                    onClick={handleSend}
                    disabled={!promptText.trim()}
                    className="p-2 text-white hover:scale-110 transition-transform disabled:opacity-50 disabled:hover:scale-100"
                >
                    <ArrowUp className="w-7 h-7" />
                </button>
            </div>
         </div>
      </div>

      {/* Icon Container */}
      <div 
        ref={containerRef}
        className="absolute bottom-10 left-0 right-0 flex flex-col items-center justify-center z-20"
        style={{ 
            opacity: iconOpacity, 
            pointerEvents: (iconOpacity < 0.1 && !isDragging) ? 'none' : 'auto',
            // Sequential Transition: Delay fade-in so it happens after prompt fades out
            // Duration: 0.3s
            transition: isDragging ? 'none' : `opacity 0.3s ease-out ${!showPrompt ? '0.3s' : '0s'}`
        }}
        onTouchStart={handleIconTouchStart}
        onTouchMove={handleIconTouchMove}
        onTouchEnd={handleIconTouchEnd}
        onMouseDown={handleIconTouchStart}
        onMouseMove={handleIconTouchMove}
        onMouseUp={handleIconTouchEnd}
        onMouseLeave={handleIconTouchEnd}
      >
        <div className="pointer-events-auto flex items-center gap-8 cursor-grab active:cursor-grabbing p-6 rounded-full">
            
            <div className="flex flex-col items-center justify-center min-w-[64px]">
                {error && (
                    <div className="mb-4 text-red-400 text-sm font-medium bg-red-900/20 px-4 py-2 rounded border border-red-900/50 backdrop-blur-sm whitespace-nowrap absolute -top-12">
                    {error}
                    </div>
                )}

                {status === 'disconnected' || status === 'error' ? (
                    <button
                    onClick={(e) => { e.stopPropagation(); onConnect(); }}
                    className="group p-2 text-white hover:opacity-80 transition-all duration-200"
                    aria-label="Start Conversation"
                    >
                    <AudioLines className="w-8 h-8 group-hover:scale-110 transition-transform" />
                    </button>
                ) : status === 'connecting' ? (
                    <button
                    disabled
                    className="p-2 text-zinc-500 cursor-not-allowed"
                    aria-label="Connecting"
                    >
                    <Loader2 className="w-8 h-8 animate-spin" />
                    </button>
                ) : (
                    <button
                    onClick={(e) => { e.stopPropagation(); onDisconnect(); }}
                    className="flex items-center justify-center w-16 h-16 bg-red-600 text-white rounded-full hover:bg-red-700 transition-all duration-200 shadow-lg shadow-red-600/20"
                    aria-label="End Session"
                    >
                    <Square className="w-5 h-5 fill-current" />
                    </button>
                )}
            </div>

            <button 
            onClick={(e) => { e.stopPropagation(); onOpenSettings(); }}
            className="w-12 h-12 rounded-full overflow-hidden hover:opacity-90 transition-opacity shadow-lg"
            >
            <img 
                src="https://avatars.githubusercontent.com/u/208861345?s=400&u=de08044480103926cedab66e3998920e83607ff7&v=4" 
                alt="Profile" 
                className="w-full h-full object-cover"
            />
            </button>

        </div>
      </div>
    </>
  );
};

export default ControlBar;
