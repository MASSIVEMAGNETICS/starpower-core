
import React, { useState } from 'react';
import { X, User, Database, Sliders, Download, LogOut, Mail, ArrowLeft, Chrome, FileText, ChevronRight, FileCode, FileSearch } from 'lucide-react';
import { TranscriptItem, GeneratedFile } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  files: GeneratedFile[];
  transcripts: TranscriptItem[];
  voicePrompt: string;
  setVoicePrompt: (prompt: string) => void;
  backgroundPrompt: string;
  setBackgroundPrompt: (prompt: string) => void;
}

type Tab = 'account' | 'data' | 'preferences' | 'files';

const SettingsModal: React.FC<SettingsModalProps> = ({ 
  isOpen, 
  onClose, 
  files, 
  transcripts,
  voicePrompt,
  setVoicePrompt,
  backgroundPrompt,
  setBackgroundPrompt
}) => {
  const [activeTab, setActiveTab] = useState<Tab>('account');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('User');
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [selectedFile, setSelectedFile] = useState<GeneratedFile | null>(null);
  
  if (!isOpen) return null;

  const getIcon = (type: string) => {
    switch (type) {
      case 'code': return <FileCode className="w-4 h-4" />;
      case 'search_report': return <FileSearch className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const handleDownloadFile = (file: GeneratedFile) => {
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-[800px] h-[550px] bg-zinc-900 border border-zinc-800 rounded-3xl shadow-2xl flex overflow-hidden animate-in fade-in zoom-in duration-200">
        
        {/* Left Sidebar - Tabs */}
        <div className="w-64 bg-zinc-950/50 border-r border-zinc-800 p-4 flex flex-col justify-between">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-white px-3 mb-4">Settings</h2>
            
            <button 
              onClick={() => setActiveTab('account')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                activeTab === 'account' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
              }`}
            >
              <User className="w-4 h-4" />
              Account
            </button>

            <button 
              onClick={() => setActiveTab('files')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                activeTab === 'files' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
              }`}
            >
              <FileText className="w-4 h-4" />
              Files
            </button>

            <button 
              onClick={() => setActiveTab('data')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                activeTab === 'data' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
              }`}
            >
              <Database className="w-4 h-4" />
              Data
            </button>

            <button 
              onClick={() => setActiveTab('preferences')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                activeTab === 'preferences' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
              }`}
            >
              <Sliders className="w-4 h-4" />
              Preferences
            </button>
          </div>

          <div className="px-3 py-2 text-xs text-zinc-600">
            v1.0.2 Beta
          </div>
        </div>

        {/* Right Content Area */}
        <div className="flex-1 flex flex-col relative bg-zinc-900">
          {/* Close Button */}
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 p-2 text-zinc-500 hover:text-white hover:bg-zinc-800 rounded-full transition-colors z-10"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Tab Content */}
          <div className="flex-1 p-0 overflow-hidden flex">

            {/* ACCOUNT TAB */}
            {activeTab === 'account' && (
              <div className="p-8 overflow-y-auto custom-scrollbar w-full">
              <div className="space-y-8 max-w-md">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">Account</h3>
                  <p className="text-zinc-500 text-sm">Manage your login and session.</p>
                </div>
                
                {!isLoggedIn ? (
                  <div className="space-y-4">
                    {!showEmailForm ? (
                        <>
                            <button 
                                onClick={() => { setIsLoggedIn(true); setUsername("Alex"); }}
                                className="w-full bg-white hover:bg-zinc-200 text-black font-medium py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-3"
                            >
                                <svg className="w-5 h-5" viewBox="0 0 24 24">
                                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.84z" fill="#FBBC05"/>
                                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                                </svg>
                                Continue with Google
                            </button>

                            <button 
                                onClick={() => setShowEmailForm(true)}
                                className="w-full bg-zinc-800 hover:bg-zinc-700 text-white font-medium py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-3"
                            >
                                <Mail className="w-5 h-5" />
                                Continue with Email
                            </button>
                        </>
                    ) : (
                        <div className="space-y-3 animate-in fade-in slide-in-from-right-4 duration-200">
                            <button 
                                onClick={() => setShowEmailForm(false)}
                                className="text-zinc-500 hover:text-white text-sm flex items-center gap-1 mb-2"
                            >
                                <ArrowLeft className="w-3 h-3" /> Back
                            </button>
                            <input 
                                type="email" 
                                placeholder="Email address" 
                                className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-zinc-600 placeholder:text-zinc-600"
                            />
                            <input 
                                type="password" 
                                placeholder="Password" 
                                className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-zinc-600 placeholder:text-zinc-600"
                            />
                             <button 
                                onClick={() => { setIsLoggedIn(true); setUsername("Alex"); }}
                                className="w-full bg-white text-black font-medium py-3 rounded-xl hover:bg-zinc-200 transition-colors mt-2"
                              >
                                Log In
                              </button>
                        </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl flex items-center gap-4">
                        <div className="w-12 h-12 bg-zinc-800 rounded-full flex items-center justify-center text-white font-bold text-lg">
                            {username[0]}
                        </div>
                        <div>
                            <h4 className="text-white font-medium">{username}</h4>
                            <p className="text-zinc-500 text-sm">alex@example.com</p>
                        </div>
                    </div>
                    
                    <div className="space-y-2">
                        <div className="flex justify-between items-center p-3 hover:bg-zinc-800/50 rounded-lg cursor-pointer transition-colors">
                            <span className="text-zinc-300 text-sm">Subscription</span>
                            <span className="text-green-400 text-xs bg-green-400/10 px-2 py-1 rounded-full">Pro Active</span>
                        </div>
                        <div className="flex justify-between items-center p-3 hover:bg-zinc-800/50 rounded-lg cursor-pointer transition-colors">
                            <span className="text-zinc-300 text-sm">Usage</span>
                            <span className="text-zinc-500 text-xs">14.2 hrs / month</span>
                        </div>
                    </div>

                    <div className="pt-4 border-t border-zinc-800">
                        <button 
                        onClick={() => { setIsLoggedIn(false); setShowEmailForm(false); }}
                        className="w-full flex items-center justify-center gap-2 text-red-400 hover:text-red-300 hover:bg-red-400/10 py-3 rounded-xl transition-colors text-sm font-medium"
                        >
                        <LogOut className="w-4 h-4" />
                        Log Out
                        </button>
                    </div>
                  </div>
                )}
              </div>
              </div>
            )}

            {/* FILES TAB */}
            {activeTab === 'files' && (
              <div className="flex w-full h-full relative">
                 {/* List View */}
                 <div 
                    className={`w-full h-full absolute inset-0 transition-all duration-300 ease-in-out flex flex-col p-8
                      ${selectedFile ? 'opacity-0 -translate-x-10 pointer-events-none' : 'opacity-100 translate-x-0'}
                    `}
                 >
                    <h3 className="text-2xl font-bold text-white mb-4">All Files</h3>
                    <div className="flex-1 overflow-y-auto custom-scrollbar space-y-2 pr-2">
                        {files.length === 0 ? (
                            <div className="text-center text-zinc-500 mt-20 px-6">
                                <FileText className="w-12 h-12 opacity-20 mx-auto mb-4" />
                                <p className="font-medium">No files yet</p>
                            </div>
                        ) : (
                            files.map((file) => (
                                <button
                                    key={file.id}
                                    onClick={() => setSelectedFile(file)}
                                    className="w-full p-3.5 bg-zinc-950 border border-zinc-800 hover:bg-zinc-800 rounded-xl flex items-center gap-4 group transition-all text-left"
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
                    className={`w-full h-full absolute inset-0 flex flex-col transition-all duration-300 ease-in-out bg-zinc-900 p-8
                      ${selectedFile ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-10 pointer-events-none'}
                    `}
                 >
                    {selectedFile && (
                        <>
                            <div className="flex items-center justify-between pb-4 border-b border-zinc-800 shrink-0 mb-4">
                                <button 
                                    onClick={() => setSelectedFile(null)}
                                    className="flex items-center text-sm font-medium text-zinc-400 hover:text-white px-3 py-2 hover:bg-zinc-800 rounded-lg transition-colors"
                                >
                                    <ChevronRight className="w-4 h-4 rotate-180 mr-1" />
                                    Back to Files
                                </button>
                                <button
                                    onClick={() => handleDownloadFile(selectedFile)}
                                    className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                                    title="Download"
                                >
                                    <Download className="w-4 h-4" />
                                </button>
                            </div>
                            <div className="flex-1 overflow-y-auto custom-scrollbar">
                                <h1 className="text-lg font-bold text-white mb-4">{selectedFile.title}</h1>
                                <pre className="whitespace-pre-wrap font-mono text-sm text-zinc-300 leading-relaxed selection:bg-zinc-700 bg-zinc-950 p-4 rounded-xl border border-zinc-800">
                                    {selectedFile.content}
                                </pre>
                            </div>
                        </>
                    )}
                 </div>
              </div>
            )}

            {/* DATA TAB */}
            {activeTab === 'data' && (
              <div className="p-8 overflow-y-auto custom-scrollbar w-full">
              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-white">Data & Storage</h3>
                
                <div className="space-y-4">
                  <div className="p-5 bg-zinc-950 border border-zinc-800 rounded-2xl flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-white">Export Chat History</h4>
                      <p className="text-sm text-zinc-500 mt-1">Download all transcripts as JSON</p>
                    </div>
                    <button 
                      onClick={() => {
                        const blob = new Blob([JSON.stringify(transcripts, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `chat_history_${Date.now()}.json`;
                        a.click();
                      }}
                      className="p-2.5 bg-zinc-800 text-white rounded-xl hover:bg-zinc-700 transition-colors"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="p-5 bg-zinc-950 border border-zinc-800 rounded-2xl flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-white">Export All Files</h4>
                      <p className="text-sm text-zinc-500 mt-1">{files.length} files generated</p>
                    </div>
                    <button 
                       onClick={() => {
                        const blob = new Blob([JSON.stringify(files, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `all_files_${Date.now()}.json`;
                        a.click();
                      }}
                      className="p-2.5 bg-zinc-800 text-white rounded-xl hover:bg-zinc-700 transition-colors"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
              </div>
            )}

            {/* PREFERENCES TAB */}
            {activeTab === 'preferences' && (
              <div className="p-8 overflow-y-auto custom-scrollbar w-full">
              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-white">Preferences</h3>
                
                <div className="space-y-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-400">System Prompt (Main Model)</label>
                    <textarea 
                      className="w-full h-32 bg-zinc-950 border border-zinc-800 rounded-xl p-4 text-sm text-zinc-300 focus:outline-none focus:border-zinc-700 resize-none custom-scrollbar"
                      placeholder="You are a helpful assistant..."
                      value={voicePrompt}
                      onChange={(e) => setVoicePrompt(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-400">Background Agent Instruction</label>
                    <textarea 
                      className="w-full h-32 bg-zinc-950 border border-zinc-800 rounded-xl p-4 text-sm text-zinc-300 focus:outline-none focus:border-zinc-700 resize-none custom-scrollbar"
                      placeholder="Instructions for the background worker..."
                      value={backgroundPrompt}
                      onChange={(e) => setBackgroundPrompt(e.target.value)}
                    />
                  </div>
                  
                  <div className="flex justify-end">
                     <button 
                      onClick={onClose}
                      className="px-4 py-2 bg-white text-black font-medium rounded-lg text-sm hover:bg-zinc-200"
                     >
                        Done
                     </button>
                  </div>
                </div>
              </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
