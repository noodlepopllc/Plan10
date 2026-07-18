import json
from pathlib import Path

class StateManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "feedback_state.json"

    def exists(self):
        return self.state_file.exists()

    def load(self):
        if not self.exists():
            return None
        with open(self.state_file, 'r') as f:
            return json.load(f)

    def save(self, state_data):
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def clear(self):
        if self.exists():
            self.state_file.unlink()