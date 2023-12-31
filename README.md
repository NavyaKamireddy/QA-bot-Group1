# QA-bot-Group1

# PROJECT PROPOSAL:
Our Task is to build a Q&A Bot over private data that answers questions about the network security course using the open-source alternatives to ChatGPT(OpenAI here) that can be run on local machine. We train the bot using network security lecture slides and network security textbook. The key idea is that once the user asks a question, the bot must be able to retrieve correct answer from lecture slides, textbook or if not available in these, it must get data from other sourcesLLMs (chatGPT here). In addition, to introduce concepts learned in our network security class we must provide traceability of the data.

# PROPOSED IDEA:
Building a Q&A bot, though challenging, can be simplified by breaking down the process into
manageable steps.
1. We have to set up the development environment involves creating a project workspace and installing relevant software and libraries.
2. We chose OpenAI from which we obtain API key. This is used before embedding. It triggers the bot to give responses from chatGPT.
3. The next big step is training. Training involves feeding the model a database of conversations so it can learn different patterns and apply them when interacting with users.
4. Once trained, the bot is ready to be integrated into chosen platform. An in depth analysis of the proposed idea are further discussed in coming sections.

# SYSTEM SPECIFICATIONS:
1) Configuration: MacBook Air, M1Chip, 8GB RAM
2) Programming Language: Python
3) Platform:JupyterNotebook(Anaconda)
4) Software Frameworks and Libraries: For appropriate deep learning framework the necessary libraries and modules imported are os, pdfreader, CharacterTextSplitter, load qa chain OpenAIEmbeddings, FAISS, OpenAI, concurrent.
5) LLM (Large Language Model): Depending on the complexity of the Q&A tasks, we used pre-trained language models which is langchain.

# SYSTEM ARCHITECTURE:
Below is the system architecture:

![image](https://github.com/NavyaKamireddy/QA-bot-Group1/assets/146391951/15547197-2ed3-464c-9d85-0398a89e0ebd)

From the above figure, we create an environment which can be accessed with API key. Once accessed the dataset is split into chunks, and then creates and saves a FAISS index with OpenAI Embeddings. This allows efficient similarity search and retrieval of relevant information from the dataset. It loads a pre-built FAISS index for document search. A prompt is defined to request succinct responses. Finally, the chatbot is executed with the input query, and the response is printed. So the steps involved are as follows:

1) Create an environment. Load the environment variable (i.e. OPENAI API KEY) from your .env file using dotenviron.
2) Initialize a ChatOpenAI instance a temperature of 0, a maximum of 50 tokens for responses, and the OpenAI API key. Default temperture is 0.7 — setting the value to 0 will reduce the randomness of ChatGPT completions.
3) Load the pre-built FAISS index using the OpenAIEmbeddings. Here after embedding each chunk is stored as a vector in database.
4) Define a prompt that includes the variable “query” and asks for answers.
5) Semantic search through the knowledge base and retrive answers.
6) If no answer found in knowledge base, retrive answers from chatGPT. For this another function is defined. It requires API key to set and get access to openAI environment. We use an engine that goes to the prompt then adjusts number of tokens to 100 extracts the answer.
7) Execute the chatbot with the formatted prompt question. Print the chatbot’s answer with the source it retrived data from.



