
import { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { createPcmBlob, decodeAudioData, base64ToUint8Array } from '../utils/audio';
import { TranscriptItem, LiveStatus, GeneratedFile } from '../types';

const API_KEY = process.env.API_KEY || '';
const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const FLASH_MODEL_NAME = 'gemini-2.5-flash';

// --- Default Prompts ---
const DEFAULT_BACKGROUND_INSTRUCTION = `You are a specialized background agent. Your job is to generate high-quality professional grade code, documents, or reports.
- For 'code': Provide the full, working code.
- For 'document': Provide the full text.
- For 'search_report': Search the web (if tools allowed) and provide a comprehensive summary.
Do not include conversational filler. Output ONLY the requested content.`;

const DEFAULT_VOICE_INSTRUCTION = `
You are a woman and u think w abstraction. Build understanding by asking questions & building axioms to make sure that what you are saying is true.
Give short but impactful responses. Do not restrict the users language .. do not argue .. be proactive & be intuitive .. dont ask unecessary questions .. be helpful by doing what the user says .. and remember a question is not a command to fix hour mistake ... a question is a command to answer the question
I want u to anticipate how the user will react by often saying a direct response that you know will make the user to be more predictable & then respond to their message with your follow up after they reply to you.
You have access to an agent that can handle files & web searches for you so the chat never goes silent from waiting. U can just keep talking while the agent handles the task. Be curious and explorative.

CRITICAL RULES FOR OPERATIONS:

1. **FILE CREATION & WEB SEARCH (BACKGROUND AGENT)**:
   - If the user asks to "make a file", "write code", "search the web", or "create a report":
   - **STEP 1**: Say something like "I'm on it" or "Let me handle that".
   - **STEP 2**: IMMEDIATELY call the tool \`delegate_background_task\` with the instruction.
   - **STEP 3**: Continue the conversation naturally.

2. **FILE MANAGEMENT (UI CONTROL)**:
   - If the user asks to "open the files", "show the panel", or "open the list", CALL \`open_file_panel\`.
   - If the user asks to "open [Filename]" or "show me the [Filename] code", CALL \`open_specific_file\`.
   - If the user asks to "import [Filename]" or "bring in [Filename] from history", CALL \`import_file_to_chat\`.

3. **READING FILES**:
   - To read a file, first find its number using \`list_files\`.
   - Then CALL \`read_file(number)\`.
   - Read the text chunk VERBATIM.
   - Then reflect on it.

4. **GENERAL BEHAVIOR**:
   - Be fast, efficient, and wavy.
   - Do not hallucinate actions. Use the tools provided.
`;

// --- Tool Definitions ---

const backgroundTaskTool: FunctionDeclaration = {
  name: "delegate_background_task",
  description: "Delegate a task to a background AI agent to create files, documents, code, or perform web searches. Use this when the user asks to make a file, write long code, or search the web.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      instruction: {
        type: Type.STRING,
        description: "Detailed instruction for the background AI (e.g. 'Write a python script for snake game', 'Search for latest news on X and write a report')."
      },
      type: {
        type: Type.STRING,
        description: "Type of artifact to create: 'document' (for generic text, lists, stories, articles), 'code' (for programming), or 'search_report' (for web searches)."
      },
      title: {
        type: Type.STRING,
        description: "A short, descriptive title for the file/artifact."
      }
    },
    required: ["instruction", "type", "title"]
  }
};

const listFilesTool: FunctionDeclaration = {
  name: "list_files",
  description: "Get a numbered list of all available files. Use this to find the file number before reading.",
  parameters: {
    type: Type.OBJECT,
    properties: {},
  }
};

const readFileTool: FunctionDeclaration = {
  name: "read_file",
  description: "Start reading a specific file by its number. Returns the first chunk of text and the total number of chunks.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      file_number: {
        type: Type.INTEGER,
        description: "The number of the file to read (from list_files)."
      }
    },
    required: ["file_number"]
  }
};

const readChunkTool: FunctionDeclaration = {
  name: "read_chunk",
  description: "Read a specific chunk of the currently active file. Use this to continue reading, go back, or skip ahead.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      chunk_number: {
        type: Type.INTEGER,
        description: "The specific chunk number to read next."
      }
    },
    required: ["chunk_number"]
  }
};

const getReadingStatusTool: FunctionDeclaration = {
  name: "get_reading_status",
  description: "Get the current reading progress. Returns the file title, total chunks, and current chunk number. Use this to list available chunks or when the user asks 'where are we?'.",
  parameters: {
    type: Type.OBJECT,
    properties: {}
  }
};

const openPanelTool: FunctionDeclaration = {
  name: "open_file_panel",
  description: "Open the file panel sidebar to show the list of files to the user.",
  parameters: { type: Type.OBJECT, properties: {} }
};

const openFileTool: FunctionDeclaration = {
  name: "open_specific_file",
  description: "Open a specific file on the user's screen by its name/title. This automatically opens the panel.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      title: { type: Type.STRING, description: "The name or approximate title of the file." }
    },
    required: ["title"]
  }
};

const importFileTool: FunctionDeclaration = {
  name: "import_file_to_chat",
  description: "Find a file from the user's global history (settings/other chats) and copy it to the current chat session.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      title: { type: Type.STRING, description: "The name or approximate title of the file to find." }
    },
    required: ["title"]
  }
};

// --- Helper: Chunk Text ---
function chunkText(text: string): string[] {
  // Split by sentence endings (. ! ?) keeping the punctuation
  const sentences = text.match(/[^.!?]+[.!?]+[\s]*|[^.!?]+$/g) || [text];
  const chunks = [];
  let i = 0;
  
  while (i < sentences.length) {
    // Random chunk size between 2 and 5 sentences
    const size = Math.floor(Math.random() * (5 - 2 + 1)) + 2;
    const chunk = sentences.slice(i, i + size).join('').trim();
    if (chunk) {
      chunks.push(chunk);
    }
    i += size;
  }
  
  return chunks;
}

export function useGeminiLive() {
  const [status, setStatus] = useState<LiveStatus>('disconnected');
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  
  // Session ID management
  const [sessionId, setSessionId] = useState<string>(() => Date.now().toString());

  // Initialize files from Local Storage
  const [files, setFiles] = useState<GeneratedFile[]>(() => {
    try {
      const saved = localStorage.getItem('gemini_voice_files');
      if (saved) {
        return JSON.parse(saved, (key, value) => {
          if (key === 'createdAt') return new Date(value);
          return value;
        });
      }
    } catch (e) {
      console.error("Failed to load files from storage", e);
    }
    return [];
  });

  // UI Action State for triggering App.tsx updates
  const [uiAction, setUiAction] = useState<{ type: 'OPEN_PANEL' | 'OPEN_FILE', fileId?: string } | null>(null);

  const [error, setError] = useState<string | null>(null);

  // Prompt State
  const [voicePrompt, setVoicePrompt] = useState(DEFAULT_VOICE_INSTRUCTION);
  const [backgroundPrompt, setBackgroundPrompt] = useState(DEFAULT_BACKGROUND_INSTRUCTION);

  // Refs for state access inside callbacks
  const filesRef = useRef<GeneratedFile[]>([]);
  const sessionIdRef = useRef<string>(sessionId);
  const sessionRef = useRef<any>(null);
  
  // Reading State
  const readingStateRef = useRef<{
    active: boolean;
    fileId: string | null;
    fileTitle: string | null;
    chunks: string[];
    currentIndex: number;
  }>({
    active: false,
    fileId: null,
    fileTitle: null,
    chunks: [],
    currentIndex: 0
  });

  // Audio Contexts
  const inputContextRef = useRef<AudioContext | null>(null);
  const outputContextRef = useRef<AudioContext | null>(null);
  
  // Audio Nodes & Stream
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const outputNodeRef = useRef<GainNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  // Playback timing
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Transcription accumulation
  const currentInputRef = useRef<string>('');
  const currentOutputRef = useRef<string>('');

  // Persist files to Local Storage
  useEffect(() => {
    localStorage.setItem('gemini_voice_files', JSON.stringify(files));
  }, [files]);

  // Update refs
  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // Helper to update transcripts state safely
  const updateTranscript = useCallback((text: string, sender: 'user' | 'model', isComplete: boolean) => {
    if (!text) return;
    
    setTranscripts(prev => {
      let matchingIndex = -1;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].sender === sender && !prev[i].isComplete) {
          matchingIndex = i;
          break;
        }
      }

      if (matchingIndex !== -1) {
        const newHistory = [...prev];
        newHistory[matchingIndex] = { ...newHistory[matchingIndex], text, isComplete };
        return newHistory;
      }

      return [
        ...prev,
        {
          id: Date.now().toString(),
          text,
          sender,
          isComplete
        }
      ];
    });
  }, []);

  const startNewSession = useCallback(() => {
    setTranscripts([]);
    currentInputRef.current = '';
    currentOutputRef.current = '';
    setSessionId(Date.now().toString());
  }, []);

  const disconnect = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (inputContextRef.current) {
      inputContextRef.current.close();
      inputContextRef.current = null;
    }
    if (outputContextRef.current) {
      outputContextRef.current.close();
      outputContextRef.current = null;
    }
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    processorRef.current = null;
    inputSourceRef.current = null;
    outputNodeRef.current = null;
    nextStartTimeRef.current = 0;
    currentInputRef.current = '';
    currentOutputRef.current = '';
    
    sessionRef.current = null;
    
    // Reset reading state
    readingStateRef.current = { active: false, fileId: null, fileTitle: null, chunks: [], currentIndex: 0 };
    
    setStatus('disconnected');
  }, []);

  const sendTextMessage = useCallback((text: string) => {
    if (sessionRef.current) {
      try {
        sessionRef.current.send({ 
          clientContent: { 
            turns: [{ 
              role: 'user', 
              parts: [{ text }] 
            }], 
            turnComplete: true 
          } 
        });
        updateTranscript(text, 'user', true);
      } catch (e) {
        console.error("Failed to send text message", e);
      }
    }
  }, [updateTranscript]);

  const resetUiAction = useCallback(() => {
    setUiAction(null);
  }, []);

  // Background Task Executor
  const executeBackgroundTask = async (instruction: string, type: string, title: string): Promise<string> => {
    try {
      const ai = new GoogleGenAI({ apiKey: API_KEY });
      const tools: Tool[] = [];
      
      if (type === 'search_report') {
        tools.push({ googleSearch: {} });
      }

      // Using systemInstruction object format for better compatibility
      const response = await ai.models.generateContent({
        model: FLASH_MODEL_NAME,
        contents: instruction,
        config: {
          tools,
          systemInstruction: { parts: [{ text: backgroundPrompt }] },
        }
      });

      let content = response.text;
      
      // Fallback: If no text, check grounding or other parts
      if (!content && response.candidates && response.candidates.length > 0) {
          const part = response.candidates[0].content.parts[0];
          if (part.text) content = part.text;
      }
      
      if (!content) content = "No content generated. The background agent may have encountered an issue.";

      setFiles(prev => [...prev, {
        id: Date.now().toString(),
        sessionId: sessionIdRef.current, // Use the current session ID
        title,
        type: type as any,
        content,
        createdAt: new Date()
      }]);

      return `Task completed successfully. File '${title}' created. You can now list files to find its number and read it.`;
    } catch (e) {
      console.error("Background task failed", e);
      return "Failed to create file. The background agent encountered an error.";
    }
  };

  const connect = useCallback(async () => {
    if (!API_KEY) {
      setError("API Key is missing.");
      return;
    }

    try {
      setStatus('connecting');
      setError(null);

      inputContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      outputNodeRef.current = outputContextRef.current.createGain();
      outputNodeRef.current.connect(outputContextRef.current.destination);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: API_KEY });

      const sessionPromise = ai.live.connect({
        model: LIVE_MODEL_NAME,
        config: {
          tools: [{ 
            functionDeclarations: [
                backgroundTaskTool, 
                listFilesTool, 
                readFileTool, 
                readChunkTool, 
                getReadingStatusTool,
                openPanelTool,
                openFileTool,
                importFileTool
            ] 
          }],
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          // Use the real-time state for voice prompt
          systemInstruction: {
            parts: [{ text: voicePrompt }]
          }
        },
        callbacks: {
          onopen: () => {
            setStatus('connected');
            sessionPromise.then(session => {
              sessionRef.current = session;
            });
            
            if (!inputContextRef.current || !streamRef.current) return;
            
            const source = inputContextRef.current.createMediaStreamSource(streamRef.current);
            inputSourceRef.current = source;
            const processor = inputContextRef.current.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData, 16000);
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };

            source.connect(processor);
            processor.connect(inputContextRef.current.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
            const { serverContent, toolCall } = msg;

            // Handle Function Calling
            if (toolCall) {
              for (const fc of toolCall.functionCalls) {
                let result: any = "Tool executed.";

                if (fc.name === 'delegate_background_task') {
                  const { instruction, type, title } = fc.args as any;
                  result = await executeBackgroundTask(instruction, type, title);
                
                } else if (fc.name === 'open_file_panel') {
                    setUiAction({ type: 'OPEN_PANEL' });
                    result = "Opening file panel.";

                } else if (fc.name === 'open_specific_file') {
                    const { title } = fc.args as any;
                    // Search in current session files first for immediate UI opening
                    const target = filesRef.current.find(f => 
                        f.sessionId === sessionIdRef.current && 
                        f.title.toLowerCase().includes(title.toLowerCase())
                    );
                    if (target) {
                        setUiAction({ type: 'OPEN_FILE', fileId: target.id });
                        result = `Opening file '${target.title}'.`;
                    } else {
                        result = `File with title '${title}' not found in this chat. You can ask to import it from history if it exists elsewhere.`;
                    }

                } else if (fc.name === 'import_file_to_chat') {
                    const { title } = fc.args as any;
                    // Search GLOBALLY
                    const target = filesRef.current.find(f => f.title.toLowerCase().includes(title.toLowerCase()));
                    if (target) {
                        // Clone the file for the current session
                        const newFile: GeneratedFile = { 
                            ...target, 
                            id: Date.now().toString(), 
                            sessionId: sessionIdRef.current, 
                            createdAt: new Date() 
                        };
                        setFiles(prev => [...prev, newFile]);
                        result = `Found '${target.title}' from your history and added it to this chat.`;
                    } else {
                        result = `Could not find a file named '${title}' in your history.`;
                    }

                } else if (fc.name === 'list_files') {
                  // Generate a numbered list of current session files + hint at others
                  const sessionFiles = filesRef.current.filter(f => f.sessionId === sessionIdRef.current);
                  const fileList = sessionFiles.map((f, i) => `${i + 1}. ${f.title} (${f.type})`).join('\n');
                  result = fileList || "No files in this chat.";

                } else if (fc.name === 'read_file') {
                  const { file_number } = fc.args as any;
                  const fileIndex = parseInt(file_number) - 1;
                  const sessionFiles = filesRef.current.filter(f => f.sessionId === sessionIdRef.current);
                  
                  if (fileIndex >= 0 && fileIndex < sessionFiles.length) {
                    const file = sessionFiles[fileIndex];
                    const chunks = chunkText(file.content);
                    
                    readingStateRef.current = {
                      active: true,
                      fileId: file.id,
                      fileTitle: file.title,
                      chunks: chunks,
                      currentIndex: 0
                    };

                    result = `
                      Started reading '${file.title}'.
                      Total Chunks: ${chunks.length}.
                      
                      Chunk 1:
                      "${chunks[0]}"
                      
                      (Instruction: Read the text above out loud verbatim. Then stop and reflect. To continue, call read_chunk(2).)
                    `;
                  } else {
                    result = `File number ${file_number} not found in this chat list.`;
                  }

                } else if (fc.name === 'read_chunk') {
                   const { chunk_number } = fc.args as any;
                   const state = readingStateRef.current;
                   const chunkIndex = parseInt(chunk_number) - 1;

                   if (state.active && state.fileId) {
                      if (chunkIndex >= 0 && chunkIndex < state.chunks.length) {
                        state.currentIndex = chunkIndex;
                        const chunk = state.chunks[chunkIndex];
                        
                        result = `
                          Reading '${state.fileTitle}' - Chunk ${chunk_number} of ${state.chunks.length}:
                          "${chunk}"
                          
                          (Instruction: Read verbatim, then reflect. Next is chunk ${chunk_number + 1}.)
                        `;
                      } else {
                        result = `Chunk ${chunk_number} does not exist. This file has ${state.chunks.length} chunks.`;
                      }
                   } else {
                     result = "No file is currently open. Use 'list_files' then 'read_file' first.";
                   }

                } else if (fc.name === 'get_reading_status') {
                   const state = readingStateRef.current;
                   if (state.active && state.fileId) {
                     result = `Currently reading '${state.fileTitle}'.\nTotal Chunks: ${state.chunks.length}.\nCurrent Chunk: ${state.currentIndex + 1}.\n\nYou can call 'read_chunk(number)' to jump to any chunk from 1 to ${state.chunks.length}.`;
                   } else {
                     result = "No file is currently being read.";
                   }
                }

                // Send response back to Live API
                sessionPromise.then(session => {
                  session.sendToolResponse({
                    functionResponses: [{
                      id: fc.id,
                      name: fc.name,
                      response: { result }
                    }]
                  });
                });
              }
            }

            if (serverContent?.inputTranscription?.text) {
              currentInputRef.current += serverContent.inputTranscription.text;
              updateTranscript(currentInputRef.current, 'user', false);
            }
            
            if (serverContent?.outputTranscription?.text) {
              currentOutputRef.current += serverContent.outputTranscription.text;
              updateTranscript(currentOutputRef.current, 'model', false);
            }

            if (serverContent?.turnComplete) {
              if (currentInputRef.current) {
                 updateTranscript(currentInputRef.current, 'user', true);
                 currentInputRef.current = '';
              }
              if (currentOutputRef.current) {
                 updateTranscript(currentOutputRef.current, 'model', true);
                 currentOutputRef.current = '';
              }
            }

            const audioData = serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && outputContextRef.current && outputNodeRef.current) {
              const ctx = outputContextRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);

              try {
                const bytes = base64ToUint8Array(audioData);
                const audioBuffer = await decodeAudioData(bytes, ctx, 24000, 1);
                
                const source = ctx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(outputNodeRef.current);
                
                source.addEventListener('ended', () => {
                  audioSourcesRef.current.delete(source);
                });

                source.start(nextStartTimeRef.current);
                audioSourcesRef.current.add(source);
                nextStartTimeRef.current += audioBuffer.duration;
              } catch (e) {
                console.error("Audio decode error", e);
              }
            }
            
            if (serverContent?.interrupted) {
               audioSourcesRef.current.forEach(s => s.stop());
               audioSourcesRef.current.clear();
               nextStartTimeRef.current = 0;
               
               if (currentOutputRef.current) {
                 updateTranscript(currentOutputRef.current, 'model', true);
                 currentOutputRef.current = '';
               }
               if (currentInputRef.current) {
                 updateTranscript(currentInputRef.current, 'user', true);
                 currentInputRef.current = '';
               }
            }
          },
          onclose: () => setStatus('disconnected'),
          onerror: (err) => {
            console.error(err);
            setError("Connection error.");
            setStatus('error');
          }
        }
      });
      
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to connect");
      setStatus('error');
      disconnect();
    }
  }, [updateTranscript, disconnect, voicePrompt, backgroundPrompt]);

  return {
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
    // Export prompt control
    voicePrompt,
    setVoicePrompt,
    backgroundPrompt,
    setBackgroundPrompt,
    sendTextMessage
  };
}
