import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Search, MessageSquare } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import axios from 'axios';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const ForensicChatbot = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Load recent messages on component mount
  useEffect(() => {
    const recentMessages: Message[] = [
      {
        id: '1',
        text: 'Welcome to the Forensic Query Interface. How can I assist with your investigation today?',
        isUser: false,
        timestamp: new Date(Date.now() - 3600000)
      },
      {
        id: '2',
        text: 'What file formats are supported for digital evidence analysis?',
        isUser: true,
        timestamp: new Date(Date.now() - 3000000)
      },
      {
        id: '3',
        text: 'We support various formats including disk images (E01, DD, RAW), mobile extractions (UFDR, TAR), and document files (PDF, DOCX, XLSX). What specific evidence type are you working with?',
        isUser: false,
        timestamp: new Date(Date.now() - 2400000)
      }
    ];
    setMessages(recentMessages);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Post to localhost backend
      const response = await axios.post('http://localhost:3001/api/query', {
        message: inputText,
        timestamp: new Date().toISOString()
      });

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.data.response || 'Response received from forensic system.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Fallback response when backend is not available
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Forensic system temporarily unavailable. Your query has been logged for analysis. Please ensure the backend service is running on localhost:3001.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);

      toast({
        title: "Connection Error",
        description: "Unable to connect to forensic backend. Check if localhost:3001 is running.",
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

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  };

  return (
    <div className="min-h-screen bg-forensic-dark p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Search className="w-8 h-8 text-forensic-glow" />
            <h1 className="text-3xl font-bold text-forensic-light">
              Forensic Query Interface
            </h1>
          </div>
          <p className="text-muted-foreground">
            Advanced digital forensics investigation assistant
          </p>
        </div>

        {/* Chat Interface */}
        <Card className="bg-card border-border shadow-2xl">
          {/* Recent Messages Header */}
          <div className="p-4 border-b border-border">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-forensic-glow" />
              <h2 className="text-lg font-semibold text-card-foreground">
                Investigation Log
              </h2>
            </div>
          </div>

          {/* Messages Area */}
          <ScrollArea className="h-96 p-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                >
                    <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                        message.isUser
                        ? 'bg-blue-100 text-blue-900' // Light blue background, dark blue text for user messages
                        : 'bg-message-bot text-card-foreground border border-border'
                    }`}
                    >
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    <span className={`text-xs mt-1 block opacity-70`}>
                      {formatTime(message.timestamp)}
                    </span>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-message-bot text-card-foreground border border-border rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-forensic-glow rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-forensic-glow rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-forensic-glow rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                      <span className="text-sm ml-2">Analyzing query...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 border-t border-border">
            <div className="flex gap-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your forensic query..."
                className="flex-1 bg-input text-foreground placeholder:text-muted-foreground border-border focus:ring-forensic-glow"
                disabled={isLoading}
              />
              <Button
                onClick={handleSendMessage}
                disabled={isLoading || !inputText.trim()}
                className="bg-primary hover:bg-primary/90 text-primary-foreground"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Connected to localhost:3001 • Press Enter to send
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ForensicChatbot;