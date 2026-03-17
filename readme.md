# AI Chat & Storytelling Architecture

This repository contains a sophisticated, web-based AI roleplay and creative writing interface powered by **Venice.ai** and **Fal.ai (Z-Image)**. It is designed for long-form storytelling, featuring automated context management, visual consistency tools, data analytics, and a dedicated "Scenario Architect."

## 🚀 Features

### 1. Advanced Chat Interface
*   **Venice.ai Integration:** Uses Venice.ai's decentralized, zero-retention API for uncensored inference.
*   **Markdown Support:** Full rendering of markdown for beautiful story formatting.
*   **Message Management:** Edit any previous message, resend user prompts, or regenerate assistant responses.
*   **Token Tracking:** Real-time tracking of input/output tokens per message and cumulative totals for the session.
*   **AI Refiner:** Send any message to Venice to automatically rewrite and purge any words or phrases present on your customizable "Banned Phrases" list. Changed text is highlighted in green/red diff blocks.

### 2. The "Scenario Architect"
A specialized meta-assistant mode designed to help you build the perfect prompt.
*   **Interview Process:** The Architect asks about genre, tone, characters, and setting.
*   **Final Draft Generation:** Once satisfied, the Architect compiles a comprehensive "System Prompt" from your discussion.
*   **One-Click Launch:** Uses the `[[SCENARIO_START]]` trigger to instantly convert your design session into an active roleplay.

### 3. Automated Summarization Engine & Memory Manager
To handle long-term memory without hitting token limits, the system includes a background summarizer:
*   **Batch Processing:** Converts older messages into concise chronological narrative paragraphs.
*   **Sliding Window:** Keeps the most recent messages as raw text for immediate continuity while referencing previous summaries for long-term context.
*   **Memory Consolidation:** Combines multiple older summary batches into a single ultra-condensed narrative archive.
*   **Multi-Turn Comparison Mode:** Whenever you restructure your memory (by consolidating or rebuilding summaries), you can enable Comparison Mode to explore two parallel timelines. Send multi-turn prompts to see how the story progresses with your new summaries vs. the original backups, tracking token consumption for each timeline independently before locking in your choice.
*   **Manual Control:** View all active summaries, regenerate specific blocks, or force a total "Rebuild".

### 4. Lorebook & Retrieval Augmented Generation (RAG)
Maintain expansive world-building documents without bloating your token context window.
*   **Vector Database:** Integrates `FAISS` and the `openai` compatible embedding endpoint to automatically chunk and embed your text lorebook into semantic vectors.
*   **Context-Aware Injection:** When you send a message, the engine calculates the semantic similarity of your text against the lorebook, dynamically retrieving only the most relevant chunks.
*   **Configurable Settings:** Adjust chunk sizing and the number of top-K elements to retrieve, ensuring the AI only receives what it needs for the current scene.
*   **Auto-Lore Extractor:** An integrated tool that lets you select a range of chat messages, batch-process them through a dedicated Venice system prompt, and automatically generate distinct, formatted encyclopedia entries to append to your Lorebook.

### 5. Chat Analytics & Insights
A robust statistical engine for tracking writing habits and model outputs:
*   **Global Memory & Projection:** See total words stored across raw text, base summaries, and consolidated archives, plus an exact projection of the token consumption for your next prompt.
*   **Data Extraction:** Specify any range of messages, filtering by User, AI, or both.
*   **N-Gram Analysis:** Instantly surfaces the most frequent words, 2-word, 3-word, and 4-word phrases.
*   **Vocabulary Metrics:** Calculates lexical richness (vocabulary diversity), average word counts, and total output size.
*   **Quality of Life:** Click any repetitive phrase in the analytics dashboard to instantly append it to your AI Refiner's "Banned Phrases" list.

### 6. Visual Memory & Image Generation
Designed for "Visual Novel" style consistency:
*   **Visual Memory Scanner:** An AI tool that scans your chat history to extract and save permanent physical descriptions of characters (hair, eyes, outfits).
*   **Context-Aware Prompts:** When generating images, the system combines the current scene description with the "Visual Memory" to ensure characters look consistent.
*   **Z-Image Integration:** Uses Fal.ai’s "Turbo" models for ultra-fast, high-quality image generation.
*   **Guided Mode:** Optional manual guidance to influence specific image generations.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   Venice.ai API Key
*   Fal.ai (FAL_KEY) API Key

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**
    The application requires API keys to function. Set them in your terminal or a `.env` file:
    ```bash
    export VENICE_API_KEY='your_venice_key_here'
    export FAL_KEY='your_fal_key_here'
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```
5.  **Access the UI:** Open `http://localhost:5000` in your browser.

---

## 📂 File Structure

*   `main.py`: The Flask backend handling API routing, file I/O, NLP analytics, vector FAISS indexing, and the summarization logic.
*   `script.js` & `index.html`: The frontend engine and UI.
*   `style.css`: Custom dark-themed styling.
*   **Configuration Files (`.json`):**
    *   `venice_settings.json`: Model selection, temperature, and penalties for chat.
    *   `summarizer_settings.json`: Logic for when and how to compress history.
    *   `image_settings.json`: Aspect ratios, HD mode, and safety toggle.
    *   `rag_settings.json`: Settings governing chunk limits and top-K embedding retrieval.
*   **System Prompts (`.txt`):**
    *   `system_prompt_main.txt`: The primary personality of the AI.
    *   `system_prompt_architect.txt`: Instructions for the scenario designer.
    *   `system_prompt_imgprompt.txt`: Logic for converting chat text into Stable Diffusion prompts.
    *   `system_prompt_visual.txt`: Instructions for extracting character traits.
    *   `system_prompt_refine.txt`: Editing instructions for Banned Phrase purging.

---

## ⚙️ Configuration

### Summarizer Logic
You can tune the summarizer in the **Settings > Summarizer** tab:
*   **Threshold:** How many messages to wait before triggering a summary.
*   **Keep Recent:** How many messages to always keep in "raw" format.
*   **Batch Size:** How many messages to compress in a single summary block.
*   **Memory Consolidation:** Configurable model and system prompt for compressing older batches.

### Image Generation
*   **Context Depth:** How many previous messages the AI looks at to determine what the image should show.
*   **Visual Memory:** Use the "Scan Context" button in settings to let the AI "learn" what your characters look like based on the story so far.

---

## ⚖️ Content Policy
The system prompts are configured for **unfiltered creative writing**. By default, the system is instructed to follow user instructions without moralizing or editorializing. Users are responsible for their own generated content and API usage.