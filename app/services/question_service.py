class QuestionService:
    def __init__(self):
        self.states = self.init_options()

    # def create_questions(self, id: str, links: list[str]) -> list[str]:
    def create_questions(self) -> list[str]:
        return_data = ["asdf"]
        return return_data

    @staticmethod
    def init_options():
        opts = "method"
        return opts

question_service = QuestionService()

def get_question_service() :
    yield question_service