
import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useGeminiLive } from './hooks/useGeminiLive';
import ControlBar from './components/ControlBar';
import LiveView from './components/LiveView';
import FilePanel from './components/FilePanel';
import SettingsModal from './components/SettingsModal';
import Sidebar from './components/Sidebar';
import { FileText, Menu } from 'lucide-react';
import { ChatSession, Project, GeneratedFile } from './types';
import { GoogleGenAI } from '@google/genai';

const App: React.FC = () => {
  const { 
    connect, 
    disconnect, 
    startNewSession, 
    status, 
    transcripts, 
    files, 
    setFiles, 
    sessionId,
    error,
    uiAction,
    resetUiAction,
    voicePrompt,
    setVoicePrompt,
    backgroundPrompt,
    setBackgroundPrompt,
    sendTextMessage
  } = useGeminiLive();

  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [panelWidth, setPanelWidth] = useState(400);
  
  // Lifted state for File Panel selection
  const [selectedFile, setSelectedFile] = useState<GeneratedFile | null>(null);

  // Track the current project context for the active session
  const [currentProjectId, setCurrentProjectId] = useState<string | undefined>(undefined);

  // Track naming status for sessions to avoid duplicate API calls
  const namingInProgressRef = useRef<Set<string>>(new Set());

  // Touch tracking for Swipe Gestures
  const touchStartXRef = useRef<number | null>(null);
  const touchStartYRef = useRef<number | null>(null);

  // Real State for Data - Initialized from Local Storage
  const [projects, setProjects] = useState<Project[]>(() => {
    try {
      const saved = localStorage.getItem('gemini_voice_projects');
      if (saved) {
        return JSON.parse(saved, (key, value) => {
          if (key === 'createdAt') return new Date(value);
          return value;
        });
      }
    } catch (e) {
      console.error("Failed to load projects from storage", e);
    }
    return [];
  });

  const [savedChats, setSavedChats] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem('gemini_voice_chats');
      if (saved) {
        return JSON.parse(saved, (key, value) => {
          if (key === 'date') return new Date(value);
          return value;
        });
      }
    } catch (e) {
      console.error("Failed to load chats from storage", e);
    }
    return [];
  });

  // Filter files for the current session
  const currentSessionFiles = useMemo(() => {
    return files.filter(f => f.sessionId === sessionId);
  }, [files, sessionId]);

  // Handle Voice UI Actions
  useEffect(() => {
    if (!uiAction) return;

    if (uiAction.type === 'OPEN_PANEL') {
        setIsPanelOpen(true);
        setIsSidebarOpen(false);
    } else if (uiAction.type === 'OPEN_FILE' && uiAction.fileId) {
        const fileToOpen = currentSessionFiles.find(f => f.id === uiAction.fileId);
        if (fileToOpen) {
            setIsPanelOpen(true);
            setIsSidebarOpen(false);
            setSelectedFile(fileToOpen);
        }
    }
    resetUiAction();
  }, [uiAction, currentSessionFiles, resetUiAction]);

  // Persistence Effects
  useEffect(() => {
    localStorage.setItem('gemini_voice_projects', JSON.stringify(projects));
  }, [projects]);

  useEffect(() => {
    localStorage.setItem('gemini_voice_chats', JSON.stringify(savedChats));
  }, [savedChats]);

  // 1. Session Initialization Effect
  // Ensures a chat entry exists in the sidebar immediately when the app starts or session changes.
  useEffect(() => {
    setSavedChats(prev => {
      const exists = prev.find(c => c.id === sessionId);
      if (exists) return prev;

      // Create new session entry immediately
      const newChat: ChatSession = {
        id: sessionId,
        title: "New Chat",
        date: new Date(),
        preview: "New conversation",
        transcripts: [],
        projectId: currentProjectId
      };
      return [newChat, ...prev];
    });
  }, [sessionId, currentProjectId]);

  // 2. Transcript Sync & Auto-Naming Effect
  // Updates the chat content in real-time and triggers naming ONLY on the first user message.
  useEffect(() => {
    if (transcripts.length === 0) return;

    const firstUserMsg = transcripts.find(t => t.sender === 'user');

    // Trigger naming if:
    // a) We have a user message
    // b) We haven't already named this session ID
    if (firstUserMsg && !namingInProgressRef.current.has(sessionId)) {
        namingInProgressRef.current.add(sessionId);
        
        // Generate title based strictly on the first message
        generateTitle(firstUserMsg.text).then(newTitle => {
            setSavedChats(prev => prev.map(chat => 
                chat.id === sessionId && chat.title === "New Chat" 
                ? { ...chat, title: newTitle.replace(/^["']|["']$/g, '') } 
                : chat
            ));
        });
    }

    // Sync transcripts to savedChats (Auto-save)
    setSavedChats(prev => prev.map(chat => {
        if (chat.id === sessionId) {
            const lastMsg = transcripts[transcripts.length - 1];
            return {
                ...chat,
                transcripts: [...transcripts],
                preview: lastMsg ? (lastMsg.text.substring(0, 40) + (lastMsg.text.length > 40 ? '...' : '')) : chat.preview
            };
        }
        return chat;
    }));
  }, [transcripts, sessionId]);

  const handleToggleSidebar = () => {
    const newState = !isSidebarOpen;
    setIsSidebarOpen(newState);
    if (newState) {
        setIsPanelOpen(false);
    }
  };

  const handleTogglePanel = () => {
    const newState = !isPanelOpen;
    setIsPanelOpen(newState);
    if (newState) {
        setIsSidebarOpen(false);
    }
  };

  const handleMainClick = () => {
    if (isSidebarOpen) {
        setIsSidebarOpen(false);
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartXRef.current = e.targetTouches[0].clientX;
    touchStartYRef.current = e.targetTouches[0].clientY;
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (touchStartXRef.current === null || touchStartYRef.current === null) return;

    const deltaX = e.changedTouches[0].clientX - touchStartXRef.current;
    const deltaY = e.changedTouches[0].clientY - touchStartYRef.current;

    // Horizontal swipe detection
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 50) {
        // Swipe Right -> Open Sidebar
        if (deltaX > 0) {
            if (!isSidebarOpen) {
                setIsSidebarOpen(true);
                setIsPanelOpen(false);
            }
        } 
        // Swipe Left -> Close Sidebar
        else {
            if (isSidebarOpen) {
                setIsSidebarOpen(false);
            }
        }
    }

    touchStartXRef.current = null;
    touchStartYRef.current = null;
  };

  // Generate a title using Gemini Flash
  const generateTitle = async (text: string): Promise<string> => {
    try {
      if (!process.env.API_KEY) return "New Conversation";
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `Generate a very short, punchy, 3-5 word title for this user message to name a chat session. Do not use quotes. User said: ${text.substring(0, 500)}`
      });
      return response.text || "New Conversation";
    } catch (e) {
      console.error("Title generation failed", e);
      return "New Conversation";
    }
  };

  const handleNewChat = (projectId?: string) => {
    // Instant switch: Just generate a new ID. 
    // The useEffects above will handle creating the list entry and saving.
    setCurrentProjectId(projectId);
    startNewSession();
    setSelectedFile(null); // Reset selected file on new chat
    if (window.innerWidth < 768) {
        setIsSidebarOpen(false);
    }
  };

  const handleCreateProject = (name: string) => {
    const newProject: Project = {
      id: Date.now().toString(),
      name,
      createdAt: new Date()
    };
    setProjects(prev => [...prev, newProject]);
  };

  const handleDeleteProject = (id: string) => {
    setProjects(prev => prev.filter(p => p.id !== id));
  };

  const handleDeleteChat = (chatId: string) => {
    setSavedChats(prev => prev.filter(c => c.id !== chatId));
    if (sessionId === chatId) {
        handleNewChat();
    }
  };

  return (
    <div className="h-screen w-screen bg-black text-white flex overflow-hidden font-sans selection:bg-zinc-800 selection:text-white relative">
      
      {/* Top Left Hamburger Menu - Hidden when sidebar OR panel is open */}
      <div className={`absolute top-6 left-6 z-50 transition-opacity duration-300 ${isSidebarOpen || isPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <button 
          onClick={handleToggleSidebar}
          className="p-2 text-white hover:opacity-70 transition-opacity"
          title="Menu"
        >
          <Menu className="w-6 h-6" />
        </button>
      </div>

      {/* Settings Modal - Passes ALL files */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
        files={files} 
        transcripts={transcripts}
        voicePrompt={voicePrompt}
        setVoicePrompt={setVoicePrompt}
        backgroundPrompt={backgroundPrompt}
        setBackgroundPrompt={setBackgroundPrompt}
      />

      {/* Sidebar (Fixed Left) */}
      <div className="z-40">
        <Sidebar 
          isOpen={isSidebarOpen} 
          onClose={() => setIsSidebarOpen(false)}
          onNewChat={handleNewChat}
          projects={projects}
          chats={savedChats}
          onCreateProject={handleCreateProject}
          onDeleteProject={handleDeleteProject}
          onDeleteChat={handleDeleteChat}
        />
      </div>

      {/* LEFT SIDE: Dedicated container for the File Panel - Passes Only SESSION Files */}
      <div 
        className={`shrink-0 h-full transition-all duration-500 cubic-bezier(0.19, 1, 0.22, 1) relative z-20`}
        style={{ 
          width: isPanelOpen ? '45vw' : '0px', 
          opacity: isPanelOpen ? 1 : 0,
          overflow: 'hidden' 
        }}
      >
        <div className="h-full w-full p-4">
          <FilePanel 
            files={currentSessionFiles} 
            isOpen={isPanelOpen} 
            onClose={() => setIsPanelOpen(false)} 
            setFiles={setFiles}
            sessionId={sessionId}
            width={panelWidth}
            setWidth={setPanelWidth}
            selectedFile={selectedFile}
            onSelectFile={setSelectedFile}
          />
        </div>
      </div>

      {/* RIGHT SIDE: Main Content Area - Resizes based on Sidebar State */}
      <main 
        onClick={handleMainClick}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        className="flex-1 relative flex flex-col h-full overflow-hidden bg-black transition-all duration-500 cubic-bezier(0.16, 1, 0.3, 1)"
        style={{
            // Push content right when sidebar opens. Sidebar is 300px + 2vh margin (approx 16px) + 8px ml. 
            marginLeft: isSidebarOpen ? '320px' : '0px',
            width: isSidebarOpen ? 'calc(100vw - 320px)' : '100%'
        }}
      >
        <LiveView transcripts={transcripts} status={status} />
        
        {/* ControlBar centered within the Right Side */}
        <ControlBar 
            status={status}
            onConnect={connect}
            onDisconnect={disconnect}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onSendText={sendTextMessage}
            error={error}
        />

        {/* File Toggle Button - Always visible, just styled differently if empty */}
        <button
            onClick={(e) => {
                e.stopPropagation();
                handleTogglePanel();
            }}
            className={`
              absolute bottom-10 left-10 z-30 w-14 h-14 bg-zinc-900 border border-zinc-800 rounded-2xl 
              flex items-center justify-center text-white hover:bg-zinc-800 transition-all shadow-lg group
              ${!isPanelOpen ? 'scale-100 opacity-100' : 'scale-90 opacity-0 pointer-events-none'}
              ${currentSessionFiles.length === 0 ? 'opacity-50 hover:opacity-100' : ''}
            `}
            title={currentSessionFiles.length === 0 ? "No files yet" : "Open Files"}
        >
            <div className="relative">
                <FileText className="w-6 h-6" />
                {currentSessionFiles.length > 0 && (
                    <span className="absolute -top-2 -right-2 w-4 h-4 bg-white text-black text-[10px] font-bold rounded-full flex items-center justify-center">
                        {currentSessionFiles.length}
                    </span>
                )}
            </div>
        </button>
      </main>
    </div>
  );
};

export default App;
