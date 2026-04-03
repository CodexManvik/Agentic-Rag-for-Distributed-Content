import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Use environment variable or fall back to localhost for development
  const BACKEND_STREAM_URL = import.meta.env.VITE_BACKEND_STREAM_URL || 'http://localhost:8000/chat/stream';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Placeholder for AI streaming message
    setMessages((prev) => [...prev, { 
      role: 'assistant', 
      content: '', 
      citations: [], 
      trace: [],
      isStreaming: true 
    }]);

    try {
      const response = await fetch(BACKEND_STREAM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (line.startsWith('data: ')) {
            try {
              const eventInfo = JSON.parse(line.substring(6));
              
              setMessages((prev) => {
                const newMessages = [...prev];
                const activeIndex = newMessages.length - 1;
                const active = { ...newMessages[activeIndex] };
                
                if (eventInfo.type === 'token') {
                  active.content += eventInfo.text;
                } else if (eventInfo.type === 'trace') {
                  active.trace = [...(active.trace || []), eventInfo.event];
                } else if (eventInfo.type === 'final') {
                  active.isStreaming = false;
                  active.citations = eventInfo.citations || [];
                  active.abstained = eventInfo.abstained;
                  active.content = active.content || eventInfo.answer || "No response text.";
                } else if (eventInfo.type === 'error') {
                  active.content += `\n[Error: ${eventInfo.message}]`;
                  active.isError = true;
                  active.isStreaming = false;
                }
                
                newMessages[activeIndex] = active;
                return newMessages;
              });
            } catch (err) {
              console.error("Parse error chunk:", err);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error fetching chat:', error);
      setMessages((prev) => {
        const newMsgs = [...prev];
        const active = { ...newMsgs[newMsgs.length - 1] };
        active.content = active.content ? active.content + "\n\n❌ Connection interrupted." : "❌ Error connecting to server.";
        active.isError = true;
        active.isStreaming = false;
        newMsgs[newMsgs.length - 1] = active;
        return newMsgs;
      });
    } finally {
      setLoading(false);
      setMessages((prev) => {
        const newMsgs = [...prev];
        if (newMsgs.length > 0) {
          const active = { ...newMsgs[newMsgs.length - 1] };
          active.isStreaming = false;
          newMsgs[newMsgs.length - 1] = active;
        }
        return newMsgs;
      });
    }
  };

  return (
    <div className="dark-theme app-container">
      <header className="chat-header">
        <h1>Smart Knowledge Navigator</h1>
        <p>Agentic RAG Assistant</p>
      </header>
      
      <div className="chat-history">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="hero-icon">✨</div>
            <h2>How can I help you today?</h2>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message-row ${msg.role}`}>
              <div className="message-icon">{msg.role === 'user' ? 'U' : 'AI'}</div>
              <div className="message-content-wrapper">
                <div className="message-content">
                  {msg.content}
                  {msg.isStreaming && <span className="cursor" />}
                </div>
                
                {msg.trace && msg.trace.length > 0 && (
                  <div className="trace-accordion">
                    <details open={msg.isStreaming}>
                      <summary>Agent Reasoning Trace <span className="trace-count">({msg.trace.length} steps)</span></summary>
                      <ul className="trace-list">
                        {msg.trace.map((t, i) => (
                           <li key={i}>
                             <span className="trace-node">[{t.node || 'Agent'}]</span> {t.status}
                             {t.duration_ms && <span className="trace-dur"> {Math.round(t.duration_ms)}ms</span>}
                           </li>
                        ))}
                        {msg.isStreaming && <li className="trace-loading">working...</li>}
                      </ul>
                    </details>
                  </div>
                )}

                {msg.abstained && <div className="warn-box">⚠️ System abstained due to insufficient evidence or policy guidelines.</div>}
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-box">
                    <strong>Sources:</strong>
                    <div className="citations-grid">
                      {msg.citations.map((cite, i) => (
                        <a key={i} href={cite.url || '#'} className="citation-pill" target="_blank" rel="noreferrer">
                          <span className="cite-idx">[{cite.index || i + 1}]</span> {cite.source_type || 'doc'}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} className="scroll-anchor" />
      </div>

      <div className="chat-input-wrapper">
        <form className="chat-input-form" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Message Smart Knowledge Navigator..."
            disabled={loading}
          />
          <button type="submit" className={input.trim() && !loading ? 'active' : ''} disabled={loading || !input.trim()}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="currentColor"/></svg>
          </button>
        </form>
        <div className="disclaimer">
          AI can make mistakes. Verify important citations.
        </div>
      </div>
    </div>
  );
}

export default App;
