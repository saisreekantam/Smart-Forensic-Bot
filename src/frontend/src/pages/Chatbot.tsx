import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Send, Search, MessageSquare, FileText, AlertTriangle, History, Plus, Trash2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Array<{
    filename: string;
    evidence_type: string;
    confidence: number;
  }>;
  confidence?: number;
}

interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  message_count: number;
}

interface CaseInfo {
  id: string;
  case_number: string;
  title: string;
  investigator_name: string;
  processed_evidence_count: number;
  total_evidence_count: number;
}

const ForensicChatbot = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [caseInfo, setCaseInfo] = useState<CaseInfo | null>(null);
  const [conversationHistory, setConversationHistory] = useState<Array<{role: string, content: string}>>([]);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      navigate('/');
      return;
    }
    fetchCaseInfo(selectedCaseId);
    loadChatSessions(selectedCaseId);
  }, [navigate]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchCaseInfo = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}`);
      if (response.ok) {
        const data = await response.json();
        setCaseInfo(data.case);
      }
    } catch (error) {
      console.error('Error fetching case info:', error);
    }
  };

  const loadChatSessions = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}/chat/sessions`);
      if (response.ok) {
        const data = await response.json();
        setChatSessions(data.sessions || []);
        
        // If no sessions exist, create a new one
        if (data.sessions.length === 0) {
          await createNewChatSession(caseId);
        } else {
          // Load the most recent session by default
          const recentSession = data.sessions[0];
          setCurrentSessionId(recentSession.id);
          await loadChatMessages(recentSession.id);
        }
      }
    } catch (error) {
      console.error('Error loading chat sessions:', error);
      await createNewChatSession(caseId);
    }
  };

  const createNewChatSession = async (caseId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/cases/${caseId}/chat/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Chat ${new Date().toLocaleDateString()}`
        })
      });

      if (response.ok) {
        const data = await response.json();
        const newSession = data.session;
        setChatSessions(prev => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
        initializeChat();
      }
    } catch (error) {
      console.error('Error creating chat session:', error);
      initializeChat(); // Fallback to local chat
    }
  };

  const loadChatMessages = async (sessionId: string) => {
    try {
      const selectedCaseId = localStorage.getItem('selectedCaseId');
      if (!selectedCaseId) return;
      
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/chat/sessions/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        const loadedMessages = data.messages.map((msg: any) => ({
          id: msg.id.toString(),
          text: msg.content,  // Use 'content' field from backend
          isUser: msg.role === 'user',
          timestamp: new Date(msg.timestamp),
          sources: msg.sources || [],
          confidence: msg.confidence
        }));
        
        if (loadedMessages.length === 0) {
          initializeChat();
        } else {
          setMessages(loadedMessages);
        }
      } else {
        initializeChat();
      }
    } catch (error) {
      console.error('Error loading chat messages:', error);
      initializeChat();
    }
  };

  const switchChatSession = async (sessionId: string) => {
    setCurrentSessionId(sessionId);
    await loadChatMessages(sessionId);
  };

  const deleteChatSession = async (sessionId: string) => {
    try {
      const selectedCaseId = localStorage.getItem('selectedCaseId');
      if (!selectedCaseId) return;
      
      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/chat/sessions/${sessionId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setChatSessions(prev => prev.filter(s => s.id !== sessionId));
        
        // If we deleted the current session, create a new one
        if (sessionId === currentSessionId) {
          const selectedCaseId = localStorage.getItem('selectedCaseId');
          if (selectedCaseId) {
            await createNewChatSession(selectedCaseId);
          }
        }
        
        toast({
          title: "Session deleted",
          description: "Chat session has been deleted successfully.",
        });
      }
    } catch (error) {
      console.error('Error deleting chat session:', error);
      toast({
        title: "Error",
        description: "Failed to delete chat session.",
        variant: "destructive"
      });
    }
  };

  const initializeChat = () => {
    const welcomeMessage: Message = {
      id: '1',
      text: 'Welcome to the ForensicAI Investigation Assistant. I can help you analyze evidence, answer questions about the case, and provide insights based on the processed data. What would you like to investigate?',
      isUser: false,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || !caseInfo) return;

    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "Error",
        description: "No case selected",
        variant: "destructive"
      });
      return;
    }

    // Check if case has processed evidence
    if (caseInfo.processed_evidence_count === 0) {
      toast({
        title: "No Evidence Processed",
        description: "Please upload and process evidence files before starting queries",
        variant: "destructive"
      });
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    
    // Update conversation history
    const newHistory = [...conversationHistory, { role: 'user', content: inputText }];
    setConversationHistory(newHistory);
    
    setInputText('');
    setIsLoading(true);

    try {
      const chatBody: any = {
        message: inputText,
        case_id: selectedCaseId,
        conversation_history: newHistory.slice(-10) // Keep last 10 messages for context
      };

      // Add session_id if available
      if (currentSessionId) {
        chatBody.session_id = currentSessionId;
      }

      const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(chatBody),
      });

      if (response.ok) {
        const data = await response.json();
        
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          isUser: false,
          timestamp: new Date(),
          sources: data.sources || [],
          confidence: data.confidence || 0
        };

        setMessages(prev => [...prev, botMessage]);
        
        // Update conversation history with bot response
        setConversationHistory(prev => [...prev, { role: 'assistant', content: data.response }]);
        
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I apologize, but I encountered an error while processing your query. Please ensure the backend service is running and try again. If the issue persists, please contact support.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);

      toast({
        title: "Connection Error",
        description: "Unable to connect to the forensic analysis backend",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (!caseInfo) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] max-w-7xl mx-auto gap-4 bg-slate-900">
      {/* Chat History Sidebar */}
      <div className={`transition-all duration-300 ${showHistory ? 'w-80' : 'w-12'} flex flex-col`}>
        <Card className="flex-1 flex flex-col bg-slate-800 border-slate-700 shadow-sm">
          <div className="p-3 border-b border-slate-700 bg-slate-800">
            <div className="flex items-center justify-between">
              {showHistory && (
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-slate-300" />
                  <span className="font-medium text-sm text-slate-200">Chat History</span>
                </div>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowHistory(!showHistory)}
                className="w-8 h-8 p-0 hover:bg-slate-700 text-slate-300"
              >
                <History className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {showHistory && (
            <>
              <div className="p-3 border-b border-slate-700">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const selectedCaseId = localStorage.getItem('selectedCaseId');
                    if (selectedCaseId) createNewChatSession(selectedCaseId);
                  }}
                  className="w-full border-slate-600 text-slate-200 hover:bg-slate-700 bg-slate-800"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  New Chat
                </Button>
              </div>

              <ScrollArea className="flex-1">
                <div className="p-2 space-y-2">
                  {chatSessions.map((session) => (
                    <div
                      key={session.id}
                      className={`group p-3 rounded-lg border cursor-pointer transition-colors ${
                        session.id === currentSessionId
                          ? 'bg-blue-900/50 border-blue-600 shadow-sm'
                          : 'border-slate-600 hover:bg-slate-700 hover:border-slate-500'
                      }`}
                      onClick={() => switchChatSession(session.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-slate-100 truncate">
                            {session.title}
                          </p>
                          <p className="text-xs text-slate-400">
                            {session.message_count} messages
                          </p>
                          <p className="text-xs text-slate-500">
                            {new Date(session.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteChatSession(session.id);
                          }}
                          className="w-6 h-6 p-0 opacity-0 group-hover:opacity-100 hover:bg-red-800 transition-opacity text-red-400"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  ))}

                  {chatSessions.length === 0 && (
                    <div className="text-center py-8 text-slate-400">
                      <History className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No chat sessions yet</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </>
          )}
        </Card>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Case Info Header */}
        <div className="bg-gradient-to-r from-blue-900 to-blue-800 text-white border rounded-lg p-4 mb-4 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <FileText className="w-5 h-5 text-blue-200" />
              <div>
                <h2 className="font-semibold text-lg text-white">{caseInfo.case_number}</h2>
                <p className="text-sm text-blue-100">{caseInfo.title}</p>
                <p className="text-xs text-blue-200">Lead: {caseInfo.investigator_name}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-blue-700 text-blue-100 border-blue-600">
                {caseInfo.processed_evidence_count}/{caseInfo.total_evidence_count} Files Processed
              </Badge>
              {caseInfo.processed_evidence_count === 0 && (
                <Badge variant="destructive" className="flex items-center gap-1 bg-red-600 text-white">
                  <AlertTriangle className="w-3 h-3" />
                  No Evidence Processed
                </Badge>
              )}
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <Card className="flex-1 flex flex-col bg-slate-800 border-slate-700 shadow-lg">
          <div className="p-4 border-b border-slate-700 bg-gradient-to-r from-slate-800 to-slate-700">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-blue-400" />
              <h3 className="font-semibold text-slate-100">ForensicAI Investigation Assistant</h3>
              <Badge variant="secondary" className="text-xs bg-slate-700 text-slate-200 border-slate-600">
                Case-Specific Analysis
              </Badge>
            </div>
            <p className="text-sm text-slate-300 mt-1">
              Ask questions about the evidence, request analysis, or seek investigative insights
            </p>
          </div>

        <ScrollArea className="flex-1 p-4 bg-slate-900" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 shadow-lg ${
                    message.isUser
                      ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white border border-blue-500'
                      : 'bg-slate-700 text-slate-100 border border-slate-600 shadow-md'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                  
                  {/* Show sources for bot messages */}
                  {!message.isUser && message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-600">
                      <p className="text-xs font-medium text-slate-300 mb-2">Sources:</p>
                      {message.sources.map((source, index) => (
                        <div key={index} className="text-xs text-slate-300 mb-1 flex items-center gap-2">
                          <FileText className="w-3 h-3 text-blue-400" />
                          <span className="font-medium">{source.filename}</span>
                          <Badge variant="outline" className="text-xs border-blue-400 text-blue-300 bg-blue-900/30">
                            {source.evidence_type}
                          </Badge>
                          {source.confidence && (
                            <span className={`font-medium ${getConfidenceColor(source.confidence)}`}>
                              {(source.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Show confidence for bot messages */}
                  {!message.isUser && message.confidence !== undefined && (
                    <div className="mt-2 pt-2 border-t border-slate-600">
                      <span className="text-xs text-slate-300">
                        Confidence: 
                        <span className={`ml-1 font-medium ${getConfidenceColor(message.confidence)}`}>
                          {(message.confidence * 100).toFixed(0)}%
                        </span>
                      </span>
                    </div>
                  )}

                  <p className={`text-xs mt-2 ${message.isUser ? 'text-blue-200' : 'text-slate-400'}`}>
                    {formatTimestamp(message.timestamp)}
                  </p>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-700 rounded-lg p-4 text-slate-100 border border-slate-600 shadow-lg max-w-[80%]">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                    <span className="text-sm">Analyzing evidence and generating response...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 border-t border-slate-700 bg-slate-800">
          {caseInfo.processed_evidence_count === 0 ? (
            <div className="flex items-center justify-center p-4 bg-amber-900/30 border border-amber-600 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-amber-400 mr-2" />
              <span className="text-sm text-amber-200">
                Upload and process evidence files before starting queries
              </span>
              <Button 
                variant="outline" 
                size="sm" 
                className="ml-4 border-amber-500 text-amber-300 hover:bg-amber-900/50"
                onClick={() => navigate('/dashboard')}
              >
                Go to Dashboard
              </Button>
            </div>
          ) : (
            <div className="flex gap-3">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about evidence, request analysis, or seek investigative insights..."
                disabled={isLoading}
                className="flex-1 border-slate-600 bg-slate-700 text-slate-100 placeholder-slate-400 focus:border-blue-500 focus:ring-blue-500 h-12 text-base px-4"
              />
              <Button 
                onClick={handleSendMessage} 
                disabled={isLoading || !inputText.trim()}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white border-0 px-6 h-12"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          )}
        </div>
        </Card>
      </div>
    </div>
  );
};

export default ForensicChatbot;