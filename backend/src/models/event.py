class Event:
    def __init__(self, id: int, name: str, date: str, description: str, impact: str):
        self.id = id
        self.name = name
        self.date = date
        self.description = description
        self.impact = impact

    def __repr__(self):
        return f"<Event(id={self.id}, name={self.name}, date={self.date}, impact={self.impact})>"