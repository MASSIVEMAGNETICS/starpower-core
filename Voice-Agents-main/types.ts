
export interface TranscriptItem {
  id: string;
  text: string;
  sender: 'user' | 'model';
  isComplete: boolean;
}

export interface AudioConfig {
  sampleRate: number;
}

export type LiveStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface GeneratedFile {
  id: string;
  sessionId?: string; // Added for scoping files to specific sessions
  title: string;
  type: 'document' | 'code' | 'search_report';
  content: string;
  createdAt: Date;
}

export interface ChatSession {
  id: string;
  title: string;
  date: Date;
  preview: string;
  projectId?: string;
  transcripts: TranscriptItem[]; // Added to store real history
}

export interface Project {
  id: string;
  name: string;
  createdAt: Date;
}
