def react (user_question,tools,max_steps=5):

    parse_action = lambda x: (x.split(" ")[0], " ".join(x.split(" ")[1:]))
    llm=[]
    history = []
    for step in range(max_steps):
        thought = llm(f"\
        wenti: {user_question}\
        lishi:{history}\
        思考下一步该用什么工具"
        )
    tool_name,tool_input = parse_action(thought)
    if tool_name == "Final Answer":
        return tool_input
    result = tools[tool_name](tool_input)
    history.append({'thought':thought,'tool':tool_name,'observation':result})

    