from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

modal = OllamaLLM(model = "mistral")

template = """

You are an expert in answering questions about a pizza restaurant. 

Here are some of the relevant reviews: {reviews}

Here are some of the questions to answers: {question}

""" 

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | modal


while True : 
    print("\n\n-----------------------------------------")
    question = input("Ask you question (q to quit) : ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)

    result =  chain.invoke({"reviews" : reviews, "question": question})
    print (result)