# DREAM CS - TUTORIAL PRICAI 2024

This repository contains code and materials for a tutorial on **Retrieval-Augmented Generation (RAG)**, **Agent**, and **Multi-Agent Systems**, presented at **PRICAI 2024**. The focus is on developing a chatbot designed to enhance attendee experience by answering questions about the PRICAI 2024 conference, using data scraped from the PRICAI website.

## Project Overview

This project demonstrates how to create a customer support chatbot using a multi-layered approach with RAG, Agents, and Multi-Agent systems. The chatbot can answer questions about PRICAI 2024, such as session schedules, speaker information, and more. By leveraging a **Retrieval-Augmented Generation** model enhanced with **Agent** and **Multi-Agent** capabilities, the chatbot provides accurate responses and handles complex queries through task specialization and interaction between multiple agents.

Key Components:
- **Retrieval-Augmented Generation (RAG)**: Efficiently retrieves relevant information for user queries.
- **Agent and Multi-Agent Systems**: Enhances the chatbot's capabilities by distributing tasks among specialized agents, improving response accuracy and efficiency.

## Setting Up the Environment

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KalbeDigitalLab/DREAM_CS-TUTORIAL-PRICAI-2024.git
   cd DREAM_CS-TUTORIAL-PRICAI-2024
   ```
2. Install required dependencies:
   Using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Configuration

1. **Create an OpenAI API Key**

   To use OpenAI's GPT-4, you need an API key. Create your key by signing up at the [OpenAI website](https://platform.openai.com/signup).

2. **Set Up the `.env` File**

   Create a `.env` file in the root directory of the project and add your OpenAI API key and any other necessary configuration variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with the API key you obtained from OpenAI.

## Project Structure

The repository has the following structure:

```
├── .gitignore
├── README.md                                     # Project README file
├── 1. RAG Customer Support.ipynb                 # Jupyter notebook for RAG setup
├── 2. Agent Customer Support.ipynb               # Jupyter notebook with Agent setup examples
├── 3. Example Multi Agent Customer Support.ipynb # Example for Multi-Agent configuration
├── multi_agent_cs_pricai2024.py                  # Main Python script for multi-agent customer support
├── requirements.txt                              # General requirements
├── requirements.yaml                             # Conda environment file
├── requirements_ubuntu.txt                       # Requirements for Ubuntu
└── utils.py                                      # Utility functions
```

## Usage and Running the Project

To run the project:

1. **Run the Notebooks**:
   - Open `RAG Customer Support.ipynb` to initialize and explore the RAG setup.
   - Use `Agent Customer Support.ipynb` for Agent setup and testing.
   - Open `Example Multi Agent Customer Support.ipynb` to understand and experiment with Multi-Agent configurations.

2. **Run the Chatbot**:
   - Execute the main script:
     ```bash
     python multi_agent_cs_pricai2024.py
     ```

3. **Interact with the Chatbot**:
   - Once running, the chatbot will be able to answer questions related to PRICAI 2024 conference content based on data from the website.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
