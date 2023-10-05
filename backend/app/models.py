from pydantic import BaseModel


class AnswerPayload(BaseModel):
    answers: list[str]
