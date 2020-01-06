def get_question_with_pos_tags(question, en_nlp):
    doc = en_nlp(u'' + question)
    question_with_pos = ""
    for token in doc:
        question_with_pos += token.text + '_' + token.pos_ + ' '
    return question_with_pos
