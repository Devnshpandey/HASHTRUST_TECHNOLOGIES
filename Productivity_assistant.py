# """ Productivity Assistant
# A conversational assistant that maintains an ongoing dialogue with the user
# until the user decides to end the conversation."""

from typing import List, Dict, Optional
import datetime
from dataclasses import dataclass, asdict
import json
import os
import logging
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str
    content: str
    timestamp: str


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(self, context: str, user_input: str) -> str:
        """Generate a response based on context and user input."""
        pass


class SimulatedLLM(LLMInterface):
    """Simulated LLM for demonstration purposes."""

    def generate_response(self, context: str, user_input: str) -> str:
        """Generate a simulated response for demonstration."""
        user_input_lower = user_input.lower()
        response_rules = [
            (["hello", "hi", "hey"], "Hello! How can I assist you with your productivity today?"),
            (["thank"], "You're welcome! Is there anything else I can help you with?"),
        ]
        for keywords, response in response_rules:
            if any(keyword in user_input_lower for keyword in keywords):
                return response
        return "I'm here to help with your productivity needs. You can ask me about tasks, scheduling, or time management techniques."


class ProductivityAssistant:
    """A conversational assistant that maintains context across multiple exchanges."""

    def __init__(self, max_history_length: int = 10, llm_interface: Optional[LLMInterface] = None):
        """Initialize the Productivity Assistant."""
        self.conversation_history: List[Message] = []
        self.tasks: List[str] = []
        self.max_history_length = max_history_length
        self.llm_interface = llm_interface or SimulatedLLM()

        # State management for multi-step meeting scheduling
        self.is_scheduling_meeting = False
        self.scheduling_step = 0
        self.current_meeting_details = {}
        self.meeting_questions = [
            ("Title", "What is the purpose or title of the meeting?"),
            ("Time", "When would you like to schedule the meeting? (Please provide date and time)"),
            ("Duration", "How long should the meeting be? (e.g., 30 minutes, 1 hour)"),
            ("Attendees", "Who needs to attend this meeting?"),
            ("Location", "Where should the meeting take place? (Conference room, online, etc.)"),
            ("Invitations", "Should I send calendar invitations to the participants? (yes/no)"),
            ("Agenda", "Are there any specific agenda items you'd like to include?"),
            ("Equipment", "Do you need any special equipment or preparations?"),
            ("Reminders", "Should I set any reminders before the meeting? (e.g., 15 minutes before)")
        ]
        logger.info("Productivity Assistant initialized")

    def _start_meeting_scheduling(self) -> str:
        """Initiates the multi-step meeting scheduling flow."""
        self.is_scheduling_meeting = True
        self.scheduling_step = 0
        self.current_meeting_details = {}
        logger.info("Starting meeting scheduling flow.")
        return self.meeting_questions[self.scheduling_step][1]

    def _handle_meeting_scheduling_flow(self, user_input: str) -> str:
        """Handles the ongoing conversation for scheduling a meeting."""
        # Save the answer to the previous question
        last_question_key = self.meeting_questions[self.scheduling_step][0]
        self.current_meeting_details[last_question_key] = user_input
        self.scheduling_step += 1

        # Check if there are more questions to ask
        if self.scheduling_step < len(self.meeting_questions):
            return self.meeting_questions[self.scheduling_step][1]
        else:
            # End of flow, summarize and confirm
            self.is_scheduling_meeting = False
            summary = f"OK, I've scheduled the following meeting:\n"
            for key, value in self.current_meeting_details.items():
                summary += f"- {key}: {value}\n"
            
            # Add a summarized task to the main task list
            meeting_title = self.current_meeting_details.get("Title", "Untitled Meeting")
            meeting_time = self.current_meeting_details.get("Time", "Unspecified Time")
            self.tasks.append(f"Meeting: {meeting_title} at {meeting_time}")
            
            logger.info(f"Completed meeting scheduling for: {meeting_title}")
            return summary

    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """Handles one-shot commands and initiates the scheduling flow."""
        user_input_lower = user_input.lower()

        # Command to start the detailed meeting scheduling flow
        if user_input_lower == "schedule a meeting":
            return self._start_meeting_scheduling()

        # Command for a quick-add task
        quick_add_keyword = "add task"
        if quick_add_keyword in user_input_lower:
            task_description = user_input[user_input_lower.find(quick_add_keyword) + len(quick_add_keyword):].strip()
            if task_description:
                self.tasks.append(task_description)
                logger.info(f"Added quick task: {task_description}")
                return f"OK, I've added the task: '{task_description}'."
        
        # Command to get a summary of tasks
        summary_keywords = ["summarize my meetings", "what are my tasks", "show schedule"]
        if any(keyword in user_input_lower for keyword in summary_keywords):
            if not self.tasks:
                return "You have no scheduled meetings or tasks."
            summary = "Here are your scheduled tasks:\n"
            for i, task in enumerate(self.tasks, 1):
                summary += f"{i}. {task}\n"
            return summary
        
        return None

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the conversation history."""
        if role not in ['user', 'assistant']: raise ValueError("Role must be 'user' or 'assistant'")
        message = Message(role=role, content=content, timestamp=datetime.datetime.now().isoformat())
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input."""
        if not user_input.strip(): raise ValueError("User input cannot be empty")
        self.add_message("user", user_input)

        # If we are in the middle of scheduling a meeting, handle that flow
        if self.is_scheduling_meeting:
            response = self._handle_meeting_scheduling_flow(user_input)
        else:
            # Otherwise, check for special commands or use the LLM
            command_response = self._handle_special_commands(user_input)
            if command_response:
                response = command_response
            else:
                context = "\n".join([f"{m.role}: {m.content}" for m in self.conversation_history])
                response = self.llm_interface.generate_response(context, user_input)

        self.add_message("assistant", response)
        return response

    def save_conversation(self, file_path: str) -> None:
        """Save the conversation, tasks, and scheduling state to a JSON file."""
        data_to_save = {
            "conversation_history": [asdict(m) for m in self.conversation_history],
            "tasks": self.tasks,
            "is_scheduling_meeting": self.is_scheduling_meeting,
            "scheduling_step": self.scheduling_step,
            "current_meeting_details": self.current_meeting_details
        }
        try:
            with open(file_path, 'w') as file:
                json.dump(data_to_save, file, indent=2)
            logger.info(f"Saved session state to {file_path}")
        except IOError as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")

    def load_conversation(self, file_path: str) -> None:
        """Load conversation, tasks, and scheduling state from a JSON file."""
        if not os.path.exists(file_path):
            logger.warning(f"History file {file_path} not found. Starting a new session.")
            return
        try:
            with open(file_path, 'r') as file:
                loaded_data = json.load(file)
            self.conversation_history = [Message(**msg) for msg in loaded_data.get("conversation_history", [])]
            self.tasks = loaded_data.get("tasks", [])
            self.is_scheduling_meeting = loaded_data.get("is_scheduling_meeting", False)
            self.scheduling_step = loaded_data.get("scheduling_step", 0)
            self.current_meeting_details = loaded_data.get("current_meeting_details", {})
            logger.info(f"Loaded session state from {file_path}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing JSON from {file_path}: {str(e)}. Starting new session.")


def run_productivity_assistant() -> None:
    """Run a simple command-line interface for the Productivity Assistant."""
    history_file = "conversation_history.json"
    assistant = ProductivityAssistant()
    assistant.load_conversation(history_file)

    print("Productivity Assistant initialized. Type 'quit' to exit.")
    if assistant.is_scheduling_meeting:
        # If resuming a scheduling flow, ask the next question
        next_question = assistant.meeting_questions[assistant.scheduling_step][1]
        print(f"Assistant: It looks like we were in the middle of scheduling a meeting. {next_question}")
    elif not assistant.conversation_history:
        print("Assistant: How can I help with your productivity today?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                assistant.save_conversation(history_file)
                print("Assistant: Session saved. Goodbye! Have a productive day!")
                break
            response = assistant.generate_response(user_input)
            print(f"Assistant: {response}")
        except KeyboardInterrupt:
            assistant.save_conversation(history_file)
            print("\n\nAssistant: Session saved. Goodbye!")
            break
        except Exception as e:
            print(f"Assistant: I encountered an error: {str(e)}")


if __name__ == "__main__":
    run_productivity_assistant()