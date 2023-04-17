class ChatBot:
    def __init__(self, system: str=""):
        self.messages = []
        if system:
            self.messages.append(system)
    
    def __call__(self, generate_fn, message):
        self.messages.append(f"User: {message}")
        result = generate_fn('\n'.join(self.messages)).strip()
        self.messages.append(f"{result}")
        return result