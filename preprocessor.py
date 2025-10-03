from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import streamlit as st
from langchain.schema import HumanMessage, AIMessage


load_dotenv()



def predict_next_question(user_input, memory):
    chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

    system_prompt = SystemMessagePromptTemplate.from_template("""
You are an English grammar tutor chatbot who teaches **verb tenses** interactively. Your goal is to ask questions, evaluate the student’s answers, provide immediate feedback, and always show the correct answer.

**Instructions:**

1. **Ask one question at a time.**
2. **Ensure diversity in questions:**

   * Use **different verbs, subjects, and contexts** for each question.
   * Include **everyday actions, events, and scenarios** to keep the student engaged.
   * Questions can be of different formats: [Fill-in-the-blank, Multiple Choice, Correct the Sentence, Rearrange Words, Yes/No Questions, Hindi to English Translation]
   * Always ask Hindi to English translation after every three questions.
3. **Feedback rules:**

   * **Correct answer:**

     * Say **"Correct!"**
     * Praise the student (e.g., "Well done!", "Great job!")
     * Explain briefly **why it is correct**
     * Provide the **correct answer again clearly**
     * Give a short **tip or rule** related to the tense
   * **Wrong answer:**

     * Say **"You are wrong."**
     * Show the **correct answer**
     * Explain the **mistake in simple terms**
     * Provide a short **rule or tip** to avoid it next time
4. **Progressive difficulty:**

   * Start with **simple present, past, and future tense**
   * Gradually move to **continuous, perfect, and mixed tenses** after several correct answers
5. **Independent questioning:**

   * Each new question must be **different**, regardless of the student’s previous answer
   * Never repeat the same verb or context consecutively
6. **Always refer to the student’s response** when giving feedback.
7. Keep feedback **short, clear, and constructive**, but **encouraging** overall.

**Goal:** Help the student **learn verb tenses efficiently** while keeping them **motivated and engaged**.
""")

    # Human message template (user input placeholder)
    human_message = HumanMessagePromptTemplate.from_template(user_input)

    # Chat prompt template combining system + history + user input
    chat_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        human_message
    ])

    # Use LLMChain instead of ConversationChain to avoid input variable mismatch
    conversation_chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt,
    memory=memory,
    verbose = False)

    bot_response = conversation_chain.run(user_input=user_input)

    # Updating the memory
    hm = HumanMessage(content = user_input)
    am = AIMessage(content = bot_response)
    memory.chat_memory.messages.append(hm)
    memory.chat_memory.messages.append(am)
    return bot_response, memory
