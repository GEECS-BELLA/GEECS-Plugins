from typing import Optional


def text_input(request: str, accepted_answers: Optional[list[str]] = None, case_insensitive: bool = True) -> str:
    while True:
        user_input = input(request)
        original = user_input

        if case_insensitive:
            if accepted_answers:
                accepted_answers = [aa.lower() for aa in accepted_answers]
            user_input = user_input.lower()

        if (not accepted_answers) or (user_input in accepted_answers):
            break

    return original
