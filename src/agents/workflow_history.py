class WorkflowHistory:
    def __init__(self, max_turns: int = 10):
        # Store the last N conversation turns
        self.max_turns = max_turns
        self.turns = []
    
    def start_turn(self, user_question: str):
        self.turns.append({
            "question": user_question,
            "steps": {}
        })

        # Keep memory within max_turns
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def add_step(self, step_name: str, content: dict):
        if not self.turns:
            return
        self.turns[-1]["steps"][step_name] = content
    
    def finalize_turn(self):
        pass
    
    def get_summary(self, k: int = 3):
        return self.turns[-k:]


        