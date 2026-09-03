import React from 'react';
import QuestionCard from '../QuestionCard/QuestionCard';
import AnswerCard from '../AnswerCard/AnswerCard';

export default function ChatWindow({ messages = [] }) {
  const chatEndRef = React.useRef(null);

  React.useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h3>💬 Interview Conversation</h3>
      </div>
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>Interview will begin here...</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            msg.role === 'ai' ? (
              <QuestionCard key={index} question={msg.content} questionNumber={msg.questionNumber} />
            ) : (
              <AnswerCard key={index} answer={msg.content} />
            )
          ))
        )}
        <div ref={chatEndRef} />
      </div>
    </div>
  );
}
