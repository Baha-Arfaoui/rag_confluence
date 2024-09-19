import os
import re
import chromadb
import gradio as gr
import json
from auth import AzureAuthenticator
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from locales.localisation import localize_prompt
from langchain.retrievers.multi_query import MultiQueryRetriever
from unidecode import unidecode
from pathlib import Path

load_dotenv()
language = os.getenv("LANGUAGE")

list_llm = [
    # "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "GPT-35",
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]


def get_llm(llm_model, temperature, max_tokens):
    # HuggingFaceHub uses HF inference endpoints

    # if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model,
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #         token=os.getenv["HF_TOKEN"]
    #     )
    if llm_model == "GPT-35":
        llm = ChatOpenAI()
    else:
        raise Exception(f"Unknown LLM: {llm_model}")
    return llm


def generate_next_questions(user_msg, llm_answer):
    llm = get_llm("GPT-35", 1, 200)
    response = llm.invoke(localize_prompt(language, "next_question_generation",
                                          user_msg=user_msg, llm_answer=llm_answer))
    questions = [[q[3:]] for q in response.content.split('\n')]
    if len(questions) != 3:
        print(f"Error: Problem with the next Question Generation: {response.content}")
        # raise Exception("Problem with the next Question Generation")
        questions = []
    return questions


def load_question(next_questions_state, example_id):
    return next_questions_state[example_id][0]


# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits, collection_name):
    embedding = OpenAIEmbeddings()
    # embedding = HuggingFaceInferenceAPIEmbeddings(
    #     api_key=os.getenv("HF_TOKEN"),
    #     api_url="https://xscl9w32v23ubm6k.eu-west-1.aws.endpoints.huggingface.cloud",
    # )
    # embedding = HuggingFaceInferenceAPIEmbeddings(
    #    api_key=os.getenv("HF_TOKEN"),
    #    api_url="https://xscl9w32v23ubm6k.eu-west-1.aws.endpoints.huggingface.cloud"
    # )

    try:
        _ = embedding.embed_query("Test test")
    except Exception as e:
        raise ConnectionResetError("The VM is currently started. Please wait a few minutes until the endpoint is live.")
    # embedding = HuggingFaceEmbeddings()
    # persist_directory = 'db'
    new_client = chromadb.EphemeralClient()

    # splitted_lists = [splits[x:x+10] for x in range(0, len(splits), 10)]
    # for list_ in splitted_lists:
    #     vectordb = Chroma.from_documents(
    #         documents=list_,
    #         embedding=embedding,
    #         # client=new_client,
    #         # collection_name=collection_name,
    #         persist_directory=persist_directory
    #     )
    # vectordb.persist()
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        # persist_directory=default_persist_directory
    )
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, vector_db, expert_prompt_name):

    llm = get_llm(llm_model, temperature, max_tokens)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever = vector_db.as_retriever()

    # Multi_query retriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )
    expert_prompt = localize_prompt(language, expert_prompt_name)
    system_template = SystemMessagePromptTemplate.from_template(localize_prompt(language, "qa_system_customer_advisor",
                                                                                slot=expert_prompt))
    human_template = HumanMessagePromptTemplate.from_template(localize_prompt(language, "qa_human_default"))

    qa_prompt_template = ChatPromptTemplate(messages=[system_template, human_template])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever_from_llm,
        chain_type="stuff",
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt_template},
        return_source_documents=True,
        # return_generated_question=False,
        verbose=False,
    )
    return qa_chain


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text
def create_collection_name(filepath):
    # Extract filename without extension
    collection_name = Path(filepath).stem
    # Fix potential issues from naming convention
    # Remove space
    collection_name = collection_name.replace(" ", "-")
    # ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    # Remove special characters
    # collection_name = re.findall("[\dA-Za-z]*", collection_name)[0]
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    # Limit length to 50 characters
    collection_name = collection_name[:50]
    # Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    # Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('Filepath: ', filepath)
    print('Collection name: ', collection_name)
    return collection_name


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    # list_file_path = [x.name for x in list_file_obj if x is not None]

    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_obj[0])
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_obj, chunk_size, chunk_overlap)
    # Create or load vector database
    progress(0.5, desc="Generating vector database...")
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"


def initialize_LLM(llm_option, llm_temperature, max_tokens, vector_db, expert_prompt_name):
    # TODO refactor
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ", llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, vector_db, expert_prompt_name)
    return qa_chain


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def conversation(message, history, expert_prompt_name):
    # TODO Do not use this as a global variable
    global local_vector_db

    formatted_chat_history = format_chat_history(message, history)
    # print("formatted_chat_history",formatted_chat_history)

    # We tested that the qa_chain creation only takes ~0.02 seconds
    exper_level_to_prompt = {

    }
    qa_chain = initialize_LLM(llm_option=0, llm_temperature=1, max_tokens=1024, vector_db=local_vector_db[0],
                              expert_prompt_name=expert_prompt_name)
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    data={"message":message,'answer':response_answer}
    file_name="data.json"
    folder_path="data_folder"
    file_path=os.path.join(folder_path,file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path,'w') as json_file :
        json.dump(data,json_file,indent=4)
    
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)

    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1]
    # next_questions_state = generate_next_questions(message, response_answer)
    # 'next_questions_state' needs to be returned twice because it needs to update the session state and the example box
    # TODO https://www.gradio.app/guides/state-in-blocks I think we can return json which would be easier to read.
    # return (gr.update(value=""), new_history, response_source1, response_source1_page, response_source2,
    #         response_source2_page, response_source3, response_source3_page, next_questions_state,
    #         next_questions_state)

    return (gr.update(value=""), new_history, response_source1, response_source1_page, response_source2,
            response_source2_page, response_source3, response_source3_page)
            
            
def return_next_question():
    file_path = "data_folder/data.json"
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    message = data["message"]
    response_answer = data["answer"]
    
    next_question_state = generate_next_questions(message, response_answer)
    
    return (next_question_state,next_question_state)

def run_both_functions(msg, chatbot, expertise_radio):
    # Call the first function and capture its return values
    msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page = conversation(msg, chatbot, expertise_radio)
    
    # Call the second function
    next_questions, next_question_box = return_next_question()
    
    # Update the outputs with the returned values from both functions
    outputs = [msg, chatbot, doc_source1, source1_page,
               doc_source2, source2_page, doc_source3, source3_page,
               next_questions, next_question_box]
    return outputs

def same_auth(username, password):
    return (username == os.getenv("USER")) and (password == os.getenv("PASSWORD"))


# The gradio reload does not work correctly. This might be part of the problem
local_vector_db = initialize_database(['Leitfaden_West_2023DIG.pdf'], 600, 200)

# Gradio Webpage | needs to be defined on the base level for gradio auto-reloading
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Starting Questions and Next Question Generation
    start_questions = [["Wer ist berechtigt für die Wohnungsbauprämie?"],
                       ["Mit wieviel Steuern muss ich bei der Riesterförderung in der Rente rechnen?"],
                       ["Welche Kosten kann ich steuerlich absetzen bei meiner vermieteten Wohnung?"]]
    next_questions = gr.State(start_questions)

    gr.set_static_paths(paths=["logo.jpg"])
    gr.Markdown("<img src='/file=logo.jpg' alt='logo_image' height='140' width='380'>")
    gr.Markdown("""<center><h2>LBS ChatBot-Demo</center></h2>
<h3>Stellen Sie beliebige Fragen zum LBS West Leitfaden 2023</h3>
<b>Anmerkung:</b> Dieser Chatbot basiert auf einem sogenannten Retrieval-Augmented-Generation Ansatz basierend auf
PDF-Dokumenten. Erstellte Antworten werden aus Referenzpassagen des Dokuments generiert und können in der
Chatinteraktion weiter verwendet werden.</i><br><br>""")
    with gr.Row():
        expertise_radio = gr.Radio(
            [("Anfänger", "expertise_beginner"),
             ("Fortgeschritten", "expertise_intermediate"),
             ("Experte", "expertise_expert")],
            label="Vorkenntnisse",
            value="expertise_beginner",
            info="Wie viel Vorwissen besitzen Sie in Bezug auf die Bankenbranche?",
        )
    with gr.Tab("Chatbot-Konversation"):

        chatbot = gr.Chatbot(height=300)

        with gr.Accordion("Top 3 Referenzen", open=False):
            with gr.Row():
                doc_source1 = gr.Textbox(label="Referenz 1 von 10", lines=2, container=True, scale=20)
                source1_page = gr.Number(label="Seite", scale=1)
            with gr.Row():
                doc_source2 = gr.Textbox(label="Referenz 2 von 10", lines=2, container=True, scale=20)
                source2_page = gr.Number(label="Seite", scale=1)
            with gr.Row():
                doc_source3 = gr.Textbox(label="Referenz 3 von 10", lines=2, container=True, scale=20)
                source3_page = gr.Number(label="Seite", scale=1)
        with gr.Row():
            msg = gr.Textbox(label='Stellen Sie eine Frage', placeholder="Nachricht schreiben", container=True, interactive=True)
        with gr.Row():
            next_question_box = gr.Dataset(samples=start_questions, components=[msg], type="index", label="Beispiele")
        with gr.Row():
            submit_btn = gr.Button("Abschicken")
            clear_btn = gr.ClearButton([msg, chatbot], value="Löschen")

        # Chatbot events
        next_question_box.click(load_question, inputs=[next_questions, next_question_box], outputs=[msg])
        msg.submit(run_both_functions,
                   inputs=[msg, chatbot, expertise_radio],
                   outputs=[msg, chatbot, doc_source1, source1_page,
                            doc_source2, source2_page, doc_source3, source3_page, next_questions, next_question_box],
                   queue=False, preprocess=False)
        submit_btn.click(run_both_functions,
                         inputs=[msg, chatbot, expertise_radio],
                         outputs=[msg, chatbot, doc_source1, source1_page,
                                  doc_source2, source2_page, doc_source3, source3_page, next_questions,
                                  next_question_box],
                         queue=False)
        
      
        
        clear_btn.click(lambda: [None, "", 0, "", 0, "", 0, start_questions, start_questions],
                        inputs=None,
                        outputs=[chatbot, doc_source1, source1_page,
                                 doc_source2, source2_page, doc_source3, source3_page, next_questions,
                                 next_question_box],
                        queue=False)


if __name__ == "__main__":
    demo.queue().launch(debug=True, auth=same_auth, allowed_paths=['logo.jpg'])
