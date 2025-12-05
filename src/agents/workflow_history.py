"""
Persistent memory for multi-turn biomedical reasoning.
"""

from typing import Any, Dict, List, Optional


class WorkflowHistory:
    """Manages conversation history with sliding window retention."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []
        self._current_turn: Optional[Dict[str, Any]] = None

    def start_turn(self, user_question: str) -> None:
        """Begin a new conversation turn."""
        self._current_turn = {
            "question": user_question,
            "steps": {},
            "entities": [],  # Store actual parsed list, not string
            "question_type": None,
        }

    def add_step(self, step_name: str, content: str) -> None:
        """Record a workflow step's output as string for logging."""
        if self._current_turn is None:
            return
        self._current_turn["steps"][step_name] = content

    def set_entities(self, entities: List[str]) -> None:

        if self._current_turn is None:
            return
        self._current_turn["entities"] = entities

    def set_question_type(self, question_type: str) -> None:
        """Store the question type for context."""
        if self._current_turn is None:
            return
        self._current_turn["question_type"] = question_type

    def finalize_turn(self) -> None:
        """Complete the current turn and add to history."""
        if self._current_turn is None:
            return
        self.turns.append(self._current_turn)
        self._current_turn = None
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_summary(self, k: int = 3) -> List[Dict[str, Any]]:
        """Get the k most recent conversation turns."""
        return self.turns[-k:] if self.turns else []

    def get_recent_entities(self, k: int = 2) -> List[str]:
        """Get flattened list of entities from k most recent turns."""
        recent = self.turns[-k:] if self.turns else []
        entities = []
        for turn in recent:
            entities.extend(turn.get("entities", []))
        # Deduplicate while preserving order
        seen = set()
        return [e for e in entities if not (e in seen or seen.add(e))]

    def get_last_question(self) -> Optional[str]:
        """Get the most recent question for context."""
        if not self.turns:
            return None
        return self.turns[-1].get("question")

    def clear(self) -> None:
        self.turns = []
        self._current_turn = None

    def __len__(self) -> int:
        return len(self.turns)
    
    def get_history_summary(self) -> str:
        """Return human-readable summary of conversation history."""
        if not self.turns:
            return "No conversation history yet."
        
        lines = []
        for i, turn in enumerate(reversed(self.turns)):
            turn_num = len(self.turns) - i
            q = turn.get("question", "")
            entities = turn.get("entities", [])
            qtype = turn.get("question_type", "unknown")
            execute_step = turn.get("steps", {}).get("execute", "no results")
            
            lines.append(f"**Turn {turn_num}:** {q}")
            lines.append(f"- Type: `{qtype}`")
            lines.append(f"- Entities: {entities if entities else 'None'}")
            lines.append(f"- Result: {execute_step}")
            lines.append("")
        
        return "\n".join(lines)