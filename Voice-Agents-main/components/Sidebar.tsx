
import React, { useState } from 'react';
import { Plus, Search, Folder, Trash2, ChevronRight, ChevronDown, MessageSquare } from 'lucide-react';
import { ChatSession, Project } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: (projectId?: string) => void;
  projects: Project[];
  chats: ChatSession[];
  onCreateProject: (name: string) => void;
  onDeleteProject: (id: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
    isOpen, 
    onClose, 
    onNewChat,
    projects,
    chats,
    onCreateProject,
    onDeleteProject
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());

  const toggleProject = (projectId: string) => {
    const newExpanded = new Set(expandedProjects);
    if (newExpanded.has(projectId)) {
        newExpanded.delete(projectId);
    } else {
        newExpanded.add(projectId);
    }
    setExpandedProjects(newExpanded);
  };

  // Filter chats based on search
  const filteredChats = chats.filter(chat => 
    chat.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    chat.preview.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group chats by project
  const unassignedChats = filteredChats.filter(chat => !chat.projectId);
  
  const handleProjectSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newProjectName.trim()) {
        onCreateProject(newProjectName.trim());
        setNewProjectName('');
        setIsCreatingProject(false);
    }
  };

  return (
    <div 
      className={`
        fixed top-0 left-0 h-[96vh] my-[2vh] ml-2 w-[300px] bg-zinc-950 border border-zinc-900 z-50
        transform transition-all duration-400 cubic-bezier(0.16, 1, 0.3, 1) flex flex-col rounded-3xl shadow-2xl
        ${isOpen ? 'translate-x-0 opacity-100' : '-translate-x-[110%] opacity-0'}
      `}
    >
      {/* Header: Search and New Chat */}
      <div className="p-5 flex items-center gap-3 shrink-0">
        <div className="relative flex-1 group">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white group-focus-within:text-white transition-colors" />
            <input 
                type="text" 
                placeholder="Search..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-zinc-900/50 border border-white/10 rounded-xl py-2.5 pl-9 pr-3 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:bg-zinc-900 focus:border-white/30 transition-all"
            />
        </div>
        <button 
            onClick={() => onNewChat()}
            className="p-2.5 text-white hover:bg-zinc-800 rounded-xl transition-colors"
            title="New Chat"
        >
            <Plus className="w-6 h-6" />
        </button>
      </div>

      {/* Separator */}
      <div className="mx-5 h-px bg-white/10 shrink-0" />

      {/* Split Container */}
      <div className="flex-1 flex flex-col overflow-hidden">
          
          {/* TOP HALF: Recent Chats */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
              <div className="p-4 space-y-1">
                  <h3 className="px-2 text-[11px] font-bold text-white uppercase tracking-widest mb-3">Recent Chats</h3>
                  {unassignedChats.length === 0 ? (
                      <p className="px-2 text-xs text-zinc-500 italic">No recent chats.</p>
                  ) : (
                      unassignedChats.map(chat => (
                          <button 
                              key={chat.id}
                              className="w-full text-left p-2.5 rounded-xl hover:bg-zinc-900/80 transition-all group flex items-start gap-3"
                          >
                              <MessageSquare className="w-4 h-4 text-white mt-0.5 shrink-0 opacity-70" />
                              <div className="min-w-0">
                                  <span className="text-sm font-medium text-white truncate block">{chat.title}</span>
                                  <span className="text-[10px] text-zinc-400 block truncate">{chat.preview}</span>
                              </div>
                          </button>
                      ))
                  )}
              </div>
          </div>

          {/* Separator */}
          <div className="mx-5 h-px bg-white/10 shrink-0" />

          {/* BOTTOM HALF: Projects */}
          <div className="flex-1 overflow-y-auto custom-scrollbar bg-black/20">
              <div className="p-4 space-y-1">
                  <div className="flex items-center justify-between px-2 mb-3">
                      <h3 className="text-[11px] font-bold text-white uppercase tracking-widest">Projects</h3>
                      <button 
                          onClick={() => setIsCreatingProject(true)}
                          className="text-white hover:opacity-70 transition-opacity"
                      >
                          <Plus className="w-4 h-4" />
                      </button>
                  </div>

                  {isCreatingProject && (
                      <form onSubmit={handleProjectSubmit} className="mb-3 px-1">
                          <input
                              autoFocus
                              type="text"
                              value={newProjectName}
                              onChange={(e) => setNewProjectName(e.target.value)}
                              onBlur={() => {
                                  if(!newProjectName.trim()) setIsCreatingProject(false);
                              }}
                              placeholder="Project Name..."
                              className="w-full bg-zinc-900 border border-white/20 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-white"
                          />
                      </form>
                  )}

                  {projects.map(project => {
                      const projectChats = filteredChats.filter(c => c.projectId === project.id);
                      const isExpanded = expandedProjects.has(project.id);

                      return (
                          <div key={project.id} className="space-y-1">
                              {/* Project Header */}
                              <div className="group flex items-center gap-2 p-2 rounded-lg hover:bg-zinc-900/50 transition-colors cursor-pointer select-none">
                                  <button 
                                      onClick={() => toggleProject(project.id)}
                                      className="p-1 text-white hover:opacity-70"
                                  >
                                      {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                                  </button>
                                  
                                  <div onClick={() => toggleProject(project.id)} className="flex-1 flex items-center gap-2 min-w-0">
                                      <Folder className="w-4 h-4 text-white" />
                                      <span className="text-sm font-medium truncate text-white">
                                          {project.name}
                                      </span>
                                  </div>

                                  <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                      <button 
                                          onClick={(e) => {
                                              e.stopPropagation();
                                              onNewChat(project.id);
                                              if (!isExpanded) toggleProject(project.id);
                                          }}
                                          className="p-1.5 text-white hover:bg-zinc-800 rounded"
                                          title="New Chat in Project"
                                      >
                                          <Plus className="w-3.5 h-3.5" />
                                      </button>
                                      <button 
                                          onClick={(e) => {
                                              e.stopPropagation();
                                              onDeleteProject(project.id);
                                          }}
                                          className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-zinc-800 rounded"
                                      >
                                          <Trash2 className="w-3.5 h-3.5" />
                                      </button>
                                  </div>
                              </div>

                              {/* Nested Chats */}
                              {isExpanded && (
                                  <div className="pl-4 space-y-0.5 border-l border-white/10 ml-3.5">
                                      {projectChats.length === 0 ? (
                                          <p className="py-2 pl-3 text-[10px] text-zinc-600">Empty project</p>
                                      ) : (
                                          projectChats.map(chat => (
                                              <button 
                                                  key={chat.id}
                                                  className="w-full text-left py-2 px-3 rounded-lg hover:bg-zinc-900 transition-all group flex items-center gap-2"
                                              >
                                                  <span className="w-1.5 h-1.5 rounded-full bg-white/50 group-hover:bg-white"></span>
                                                  <span className="text-xs text-zinc-400 group-hover:text-white truncate">{chat.title}</span>
                                              </button>
                                          ))
                                      )}
                                  </div>
                              )}
                          </div>
                      );
                  })}
              </div>
          </div>
      </div>
    </div>
  );
};

export default Sidebar;
