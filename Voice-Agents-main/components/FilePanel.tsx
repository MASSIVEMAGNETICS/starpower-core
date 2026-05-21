
import React, { useState, useRef, useEffect } from 'react';
import { FileText, X, Download, ChevronRight, FileCode, FileSearch, Upload, Edit2, Check, GripVertical, Maximize2, Minimize2 } from 'lucide-react';
import { GeneratedFile } from '../types';

interface FilePanelProps {
  files: GeneratedFile[];
  isOpen: boolean;
  onClose: () => void;
  setFiles: React.Dispatch<React.SetStateAction<GeneratedFile[]>>;
  sessionId: string;
  width: number;
  setWidth: (width: number) => void;
  selectedFile: GeneratedFile | null;
  onSelectFile: (file: GeneratedFile | null) => void;
}

const FilePanel: React.FC<FilePanelProps> = ({ 
  files, 
  isOpen, 
  onClose, 
  setFiles, 
  sessionId, 
  width, 
  setWidth,
  selectedFile,
  onSelectFile
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [isMobile, setIsMobile] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Check for mobile on mount and resize
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Sync edit content when selected file changes externally
  useEffect(() => {
    if (selectedFile) {
        setEditContent(selectedFile.content);
    } else {
        setIsEditing(false);
    }
  }, [selectedFile]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'code': return <FileCode className="w-5 h-5" />;
      case 'search_report': return <FileSearch className="w-5 h-5" />;
      default: return <FileText className="w-5 h-5" />;
    }
  };

  const handleDownload = (file: GeneratedFile) => {
    const blob = new Blob([file.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file.title.replace(/\s+/g, '_')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadAll = () => {
    const blob = new Blob([JSON.stringify(files, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_files_${sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      const newFile: GeneratedFile = {
        id: Date.now().toString(),
        sessionId: sessionId, // Tag with current session ID
        title: file.name,
        type: 'document',
        content: content,
        createdAt: new Date()
      };
      setFiles(prev => [...prev, newFile]);
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  const startEditing = (file: GeneratedFile) => {
    setIsEditing(true);
    setEditContent(file.content);
  };

  const saveEdit = () => {
    if (selectedFile) {
      const updatedFile = { ...selectedFile, content: editContent };
      setFiles(prev => prev.map(f => f.id === selectedFile.id ? updatedFile : f));
      onSelectFile(updatedFile);
      setIsEditing(false);
    }
  };

  return (
    <>
      <div 
        ref={panelRef}
        className={`
          h-full w-full
          bg-zinc-950 flex flex-col overflow-hidden
          rounded-3xl border border-zinc-800 shadow-2xl
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800/50 shrink-0">
          <h2 className="text-lg font-medium text-white flex items-center gap-2">
              Files
              <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full">{files.length}</span>
          </h2>
          <div className="flex items-center gap-1">
              <button
                  onClick={handleDownloadAll}
                  className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Download All Files"
              >
                  <Download className="w-4 h-4" />
              </button>
              <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Upload File"
              >
                  <Upload className="w-4 h-4" />
              </button>
              <button 
                  onClick={onClose}
                  className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
              >
                  <X className="w-4 h-4" />
              </button>
          </div>
          <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept=".txt,.md,.js,.ts,.py,.json,.html,.css" 
              onChange={handleFileUpload}
          />
        </div>

        {/* Content Container */}
        <div className="flex-1 flex overflow-hidden relative w-full">
          {/* List View */}
          <div 
            className={`w-full h-full absolute inset-0 transition-all duration-300 ease-in-out flex flex-col
              ${selectedFile ? 'opacity-0 -translate-x-10 pointer-events-none' : 'opacity-100 translate-x-0'}
            `}
          >
              <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
                  {files.length === 0 ? (
                      <div className="text-center text-zinc-500 mt-20 px-6">
                          <div className="w-12 h-12 bg-zinc-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                              <FileText className="w-6 h-6 opacity-50" />
                          </div>
                          <p className="font-medium">No files yet</p>
                          <p className="text-sm mt-1 opacity-70">Ask the AI to create documents, code, or reports.</p>
                      </div>
                  ) : (
                      files.map((file) => (
                          <button
                              key={file.id}
                              onClick={() => onSelectFile(file)}
                              className="w-full p-3.5 bg-zinc-900/30 border border-zinc-800/50 hover:bg-zinc-800/50 hover:border-zinc-700/50 rounded-xl flex items-center gap-4 group transition-all text-left"
                          >
                              <div className={`p-2.5 rounded-lg transition-colors ${
                                file.type === 'code' ? 'bg-blue-500/10 text-blue-400' : 
                                file.type === 'search_report' ? 'bg-emerald-500/10 text-emerald-400' : 
                                'bg-zinc-800 text-zinc-400 group-hover:text-white'
                              }`}>
                                  {getIcon(file.type)}
                              </div>
                              <div className="flex-1 min-w-0">
                                  <h3 className="font-medium text-zinc-200 truncate text-sm">{file.title}</h3>
                                  <p className="text-[11px] text-zinc-500 mt-0.5 capitalize">{file.type} • {file.createdAt.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</p>
                              </div>
                              <ChevronRight className="w-4 h-4 text-zinc-700 group-hover:text-zinc-400 transition-colors" />
                          </button>
                      ))
                  )}
              </div>
          </div>

          {/* Detail View */}
          <div 
            className={`w-full h-full absolute inset-0 flex flex-col transition-all duration-300 ease-in-out bg-zinc-900/50
              ${selectedFile ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-10 pointer-events-none'}
            `}
          >
              {selectedFile && (
                  <>
                      <div className="flex items-center justify-between p-2 border-b border-zinc-800/50 shrink-0">
                          <button 
                              onClick={() => {
                                  onSelectFile(null);
                                  setIsEditing(false);
                              }}
                              className="flex items-center text-xs font-medium text-zinc-400 hover:text-white px-2 py-1.5 hover:bg-zinc-800 rounded-lg transition-colors"
                          >
                              <ChevronRight className="w-3.5 h-3.5 rotate-180 mr-1" />
                              Back
                          </button>
                          <div className="flex items-center gap-1">
                              {isEditing ? (
                                  <button
                                      onClick={saveEdit}
                                      className="p-1.5 text-green-400 hover:text-green-300 hover:bg-green-400/10 rounded-lg flex items-center gap-1.5 transition-colors"
                                      title="Save Changes"
                                  >
                                      <Check className="w-3.5 h-3.5" />
                                      <span className="text-xs font-bold">Save</span>
                                  </button>
                              ) : (
                                  <button
                                      onClick={() => startEditing(selectedFile)}
                                      className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                                      title="Edit Content"
                                  >
                                      <Edit2 className="w-3.5 h-3.5" />
                                  </button>
                              )}
                              <button
                                  onClick={() => handleDownload(selectedFile)}
                                  className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                                  title="Download"
                              >
                                  <Download className="w-3.5 h-3.5" />
                              </button>
                          </div>
                      </div>
                      <div className="flex-1 overflow-hidden relative">
                          {isEditing ? (
                              <textarea
                                  className="w-full h-full bg-zinc-950/50 text-zinc-300 font-mono text-sm p-4 resize-none focus:outline-none focus:ring-1 focus:ring-zinc-700"
                                  value={editContent}
                                  onChange={(e) => setEditContent(e.target.value)}
                                  spellCheck={false}
                              />
                          ) : (
                               <div className="absolute inset-0 overflow-y-auto p-4 custom-scrollbar">
                                  <h1 className="text-lg font-bold text-white mb-4 sticky top-0 bg-zinc-900/95 backdrop-blur-md pb-2 border-b border-zinc-800 z-10">{selectedFile.title}</h1>
                                  <pre className="whitespace-pre-wrap font-mono text-sm text-zinc-300 leading-relaxed selection:bg-zinc-700">
                                      {selectedFile.content}
                                  </pre>
                               </div>
                          )}
                      </div>
                  </>
              )}
          </div>
        </div>
      </div>
    </>
  );
};

export default FilePanel;
