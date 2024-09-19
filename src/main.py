
if __name__ == '__main__':
    from help_desk import HelpDesk

    model = HelpDesk(new_db=True)

    print(model.db._collection.count())

    prompt = 'Quelle est la différence entre un data scientist et un data engineer ?'
    result, sources = model.retrieval_qa_inference(prompt)
    print(result, sources)
