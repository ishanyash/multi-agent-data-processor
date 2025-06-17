import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import uuid

class Message:
    def __init__(self, sender: str, receiver: str, message_type: str, content: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class MessageBroker:
    def __init__(self):
        self.message_queue = []
        self.subscribers = {}
        
    def publish(self, message: Message):
        """Publish a message (simplified synchronous version)"""
        self.message_queue.append(message)
        
    def subscribe(self, agent_name: str, callback):
        """Subscribe an agent to receive messages"""
        self.subscribers[agent_name] = callback
        
    def process_messages(self):
        """Process all messages in the queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            if message.receiver in self.subscribers:
                self.subscribers[message.receiver](message)
